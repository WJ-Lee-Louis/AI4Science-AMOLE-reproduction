import os
import datetime
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from utils.util import cycle_index, save

# Call dataset
from datasets import TanimotoSTM_Datasets_Graph
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.argument import config2string
from utils.bert import prepare_text_tokens

# For Language Models
from transformers import AutoModel, AutoTokenizer

# For Graph Neural Networks
from layers import GNN, GNN_graphpred

class trainer:
    
    def __init__(self, args):
        
        self.args = args
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

        d = datetime.datetime.now()
        date = d.strftime("%x")[-2:] + d.strftime("%x")[0:2] + d.strftime("%x")[3:5]

        self.config_str = "{}_".format(date) + config2string(args)
        if self.rank == 0:
            print("\n[Config] {}\n".format(self.config_str))
                
        os.environ['TOKENIZERS_PARALLELISM'] = 'False'

        # Select GPU device
        if not torch.cuda.is_available():
            raise RuntimeError("AMOLE pretraining requires a CUDA-capable GPU.")
        if args.device < 0 or args.device >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid logical CUDA device {args.device}; "
                f"visible device count is {torch.cuda.device_count()}."
            )
        self.device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(args.device)
        
        ##### Load Train Data #####
        dataloader_class = pyg_DataLoader
        if args.dataset == "TanimotoSTM":
            self.dataset = TanimotoSTM_Datasets_Graph(
                args.data_path,
                aug=args.p_aug,
                num_cand=args.num_cand,
                augmentation_strategy=args.augmentation_strategy,
                curriculum_start_k=args.curriculum_start_k,
                curriculum_warmup_epochs=args.curriculum_warmup_epochs,
                curriculum_rank_increment=args.curriculum_rank_increment,
                stratified_min_similarity=args.stratified_min_similarity,
                stratified_high_probability=args.stratified_high_probability,
                stratified_mid_probability=args.stratified_mid_probability,
                stratified_low_probability=args.stratified_low_probability,
            )
        else:
            raise Exception

        self.sampler = None
        if self.distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )

        generator = torch.Generator()
        generator.manual_seed(args.seed + self.rank)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.dataloader = dataloader_class(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=args.num_workers,
            drop_last=self.distributed,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        if self.rank == 0 and self.distributed:
            print(
                f"[Distributed] world_size={self.world_size}, "
                f"local_batch={args.batch_size}, "
                f"global_batch={args.batch_size * self.world_size}, "
                f"steps_per_epoch={len(self.dataloader)}"
            )

    
    def build_LM(self, args):
        """
        Build Language Models for Encoding Textual Description
        """
        if args.lm == "SciBERT":
            pretrained_SciBERT_folder = os.path.join(args.data_path, 'pretrained_SciBERT')
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_SciBERT_folder,
                local_files_only=True,
            )
            self.text_model = AutoModel.from_pretrained(
                pretrained_SciBERT_folder,
                local_files_only=True,
            ).to(self.device)
            if args.gradient_checkpointing:
                self.text_model.gradient_checkpointing_enable()
            self.text_dim = 768
        else:
            raise Exception


    def build_GNN(self, args):
        """
        Build Graph Neural Networks for Encoding Molecular Graph Structure
        """
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)

        self.molecule_model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)

        pretrained_model_path = os.path.join(args.data_path, "pretrained_GraphMVP", args.pretrain_gnn_mode, "model.pth")
        self.molecule_model.from_pretrained(pretrained_model_path, self.device)
        self.molecule_model.to(self.device)
        self.molecule_dim = args.gnn_emb_dim
    

    def get_text_repr(self, text):
        """
        Get representation of molecular textual description with Language Models
        """
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=self.device,
            description=text,
            tokenizer=self.text_tokenizer,
            max_seq_len=self.args.max_seq_len,
            dynamic_padding=self.args.dynamic_padding,
        )
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)
        
        return text_repr
    

    def get_molecule_repr(self, molecule):
        """
        Get representation of molecules with Graph Neural Networks
        """        
        molecule_output, _ = self.molecule_model(molecule.to(self.device))
        molecule_repr = self.mol2latent(molecule_output)

        return molecule_repr
    

    def calc_S2P_Loss(self, X, Y, Tanimoto_mat):
        """
        Calculate the Contrastive Loss for Model Training
        """
            
        Tanimoto_mat = torch.div(Tanimoto_mat, self.args.target_T)
        soft_label = F.softmax(Tanimoto_mat, dim = 1)

        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, self.args.T)
        logprobs = F.log_softmax(logits, dim = 1)

        loss = - (soft_label * logprobs).sum() / logits.shape[0]
        
        return loss


    def save_model(self, epoch = None):

        if self.rank != 0:
            return

        def unwrap(model):
            return model.module if hasattr(model, "module") else model

        save(self.args.checkpoint_path, "text", unwrap(self.text_model), self.config_str)
        save(self.args.checkpoint_path, "molecule", unwrap(self.molecule_model), self.config_str)

        if self.text2latent is not None:
            save(self.args.checkpoint_path, "text2latent", unwrap(self.text2latent), self.config_str)
        
        if self.mol2latent is not None:
            save(self.args.checkpoint_path, "mol2latent", unwrap(self.mol2latent), self.config_str)
