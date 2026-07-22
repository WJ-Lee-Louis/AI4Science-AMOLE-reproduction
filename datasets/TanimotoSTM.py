import sys
sys.path.append('.')

import os
from itertools import repeat
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle
import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class TanimotoSTM_Datasets_Graph(InMemoryDataset):
    def __init__(
        self,
        root,
        aug=0.0,
        num_cand=30,
        augmentation_strategy="baseline",
        curriculum_start_k=10,
        curriculum_warmup_epochs=5,
        curriculum_rank_increment=4,
        stratified_min_similarity=0.25,
        stratified_high_probability=0.50,
        stratified_mid_probability=0.35,
        stratified_low_probability=0.15,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")
        # `similarity` result file
        self.similarity_file_path = os.path.join(self.root, "processed/similarities_CID.pt")
        self.similarity_score_file_path = os.path.join(
            self.root, "processed/similarity_scores_CID.pt"
        )
        self.same_cid_path = os.path.join(self.root, "processed/same_CID.pt")
        self.p_aug = aug
        self.num_cand = num_cand
        self.augmentation_strategy = augmentation_strategy
        self.curriculum_start_k = curriculum_start_k
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.curriculum_rank_increment = curriculum_rank_increment
        self.stratified_min_similarity = stratified_min_similarity
        self.stratified_group_probabilities = np.asarray(
            [
                stratified_high_probability,
                stratified_mid_probability,
                stratified_low_probability,
            ],
            dtype=np.float64,
        )
        self.current_epoch = 1

        if self.augmentation_strategy not in {"baseline", "curriculum", "stratified"}:
            raise ValueError(
                f"Unknown augmentation strategy: {self.augmentation_strategy}"
            )
        if not 1 <= self.curriculum_start_k <= self.num_cand:
            raise ValueError("curriculum_start_k must be between 1 and num_cand")
        if self.curriculum_warmup_epochs < 1:
            raise ValueError("curriculum_warmup_epochs must be positive")
        if self.curriculum_rank_increment < 1:
            raise ValueError("curriculum_rank_increment must be positive")
        if self.augmentation_strategy == "stratified":
            if self.num_cand < 3:
                raise ValueError("stratified augmentation requires num_cand >= 3")
            if not 0.0 <= self.stratified_min_similarity <= 1.0:
                raise ValueError("stratified_min_similarity must be between 0 and 1")
            if np.any(self.stratified_group_probabilities < 0):
                raise ValueError("stratified group probabilities must be non-negative")
            if not np.isclose(self.stratified_group_probabilities.sum(), 1.0):
                raise ValueError("stratified group probabilities must sum to 1")

        super(TanimotoSTM_Datasets_Graph, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.load_Graph_CID_and_text()

        return

    def set_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("epoch must be one-indexed and positive")
        self.current_epoch = epoch

    def get_candidate_count(self, epoch=None):
        if self.augmentation_strategy in {"baseline", "stratified"}:
            return self.num_cand

        epoch = self.current_epoch if epoch is None else epoch
        if epoch <= self.curriculum_warmup_epochs:
            return self.curriculum_start_k

        expanded_k = self.curriculum_start_k + self.curriculum_rank_increment * (
            epoch - self.curriculum_warmup_epochs
        )
        return min(expanded_k, self.num_cand)

    def get_stratified_rank_groups(self):
        """Return high/mid/low rank groups split at 20% and 80%."""
        high_end = max(1, int(np.ceil(self.num_cand * 0.20)))
        low_start = min(self.num_cand - 1, int(np.ceil(self.num_cand * 0.80)))
        ranks = np.arange(self.num_cand)
        return ranks[:high_end], ranks[high_end:low_start], ranks[low_start:]

    def sample_augmentation_cid(self, CID):
        """Sample a replacement CID, or return None to retain the original."""
        if np.random.binomial(1, self.p_aug) == 0:
            return None

        candidate_count = self.get_candidate_count()
        similar_CIDs = np.asarray(self.similarity[CID][:candidate_count])
        if len(similar_CIDs) != candidate_count:
            raise RuntimeError(
                f"CID {CID} has {len(similar_CIDs)} candidates; expected {candidate_count}"
            )

        if self.augmentation_strategy != "stratified":
            return int(np.random.choice(similar_CIDs))

        similarity_scores = np.asarray(
            self.similarity_scores[CID][:candidate_count], dtype=np.float32
        )
        if len(similarity_scores) != candidate_count:
            raise RuntimeError(
                f"CID {CID} has {len(similarity_scores)} similarity scores; "
                f"expected {candidate_count}"
            )

        rank_groups = self.get_stratified_rank_groups()
        selected_group = int(
            np.random.choice(len(rank_groups), p=self.stratified_group_probabilities)
        )
        group_ranks = rank_groups[selected_group]
        eligible_ranks = group_ranks[
            similarity_scores[group_ranks] >= self.stratified_min_similarity
        ]
        if len(eligible_ranks) == 0:
            return None
        selected_rank = int(np.random.choice(eligible_ranks))
        return int(similar_CIDs[selected_rank])

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        suppl = Chem.SDMolSupplier(self.SDF_file_path)

        CID2graph = {}
        for mol in tqdm(suppl):
            CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)

            # Create Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            fp = torch.tensor(array).reshape(1, -1)
            graph.fp = fp
            
            CID2graph[CID] = graph
        print("CID2graph", len(CID2graph))
        
        with open(self.CID2text_file, "r") as f:
            CID2text_data = json.load(f)
        print("CID2data", len(CID2text_data))
            
        CID_list, graph_list, text_list = [], [], []
        for CID, value_list in CID2text_data.items():
            CID = int(CID)
            if CID not in CID2graph:
                print("CID {} missing".format(CID))
                continue
            graph = CID2graph[CID]
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
                graph_list.append(graph)

        CID_text_df = pd.DataFrame({"CID": CID_list, "text": text_list})
        CID_text_df.to_csv(self.CID_text_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])

        return
    
    def load_Graph_CID_and_text(self):
        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID_text_file_path)
        self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()

        # Load similar molecules
        self.similarity = torch.load(self.similarity_file_path)
        self.similarity_scores = None
        if self.augmentation_strategy == "stratified":
            self.similarity_scores = torch.load(self.similarity_score_file_path)
            if set(self.similarity) != set(self.similarity_scores):
                raise RuntimeError(
                    "Similarity candidate and score artifacts have different CID keys"
                )

        # Multiple descriptions can share a CID. Any occurrence has the same graph,
        # so retaining the first index avoids an O(dataset size) list scan per sample.
        self.CID_to_first_index = {}
        for index, CID in enumerate(self.CID_list):
            self.CID_to_first_index.setdefault(CID, index)

        self.same_CID = torch.load(self.same_cid_path)

        self.CID_key_list = list()
        for i in range(len(self.same_CID)):
            if len(self.same_CID[i]) > 0:
                self.CID_key_list.append(i)

        return


    def get(self, idx):
        text = self.text_list[idx]

        aux_idx = np.random.choice(self.CID_key_list)
        aux_text = self.text_list[aux_idx]
        sameCIDidx = np.random.choice(self.same_CID[aux_idx])
        new_text = self.text_list[sameCIDidx]
        aux_text += " [SEP] " + new_text

        CID = self.CID_list[idx]
        similar_CID = self.sample_augmentation_cid(CID)
        similar_index = (
            self.CID_to_first_index[similar_CID] if similar_CID is not None else idx
        )

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(
                slices[similar_index], slices[similar_index + 1]
            )
            data[key] = item[s]
        
        original_data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            original_data[key] = item[s]

        return text, original_data, data, aux_text

    def __len__(self):
        return len(self.text_list)



if __name__ == "__main__":
    
    DATA_PATH = "./data/PubChemSTM"
    batch_size = 45
    num_workers = 6
    
    dataset = TanimotoSTM_Datasets_Graph(DATA_PATH)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")
