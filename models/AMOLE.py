import sys
sys.path.append('.')

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel
from contextlib import ExitStack

import copy

# Call Trainer class
from trainer import trainer

from utils.util import freeze_network, cycle_index
from utils.bert import prepare_text_tokens_kd
from utils.chem import calc_tanimoto
from utils import argument


class AMOLE_Trainer(trainer):

    def __init__(self, args):
        trainer.__init__(self, args)

        ##### Build Language Model and Graph Neural Networks #####
        self.build_LM(args)
        self.build_GNN(args)
        
        self.text2latent = nn.Linear(self.text_dim, args.SSL_emb_dim).to(self.device)
        self.mol2latent = nn.Linear(self.molecule_dim, args.SSL_emb_dim).to(self.device)

        # Freeze Teacher Network
        self.ttext2latent = copy.deepcopy(self.text2latent)
        self.ttext_model = copy.deepcopy(self.text_model)
        freeze_network(self.ttext2latent)
        freeze_network(self.ttext_model)
        self.distill_loss = nn.MSELoss()

        if self.distributed:
            self.molecule_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.molecule_model)
            ddp_kwargs = {
                "device_ids": [args.device],
                "output_device": args.device,
                "broadcast_buffers": True,
                "gradient_as_bucket_view": True,
            }
            self.text_model = DistributedDataParallel(self.text_model, **ddp_kwargs)
            self.molecule_model = DistributedDataParallel(
                self.molecule_model,
                find_unused_parameters=True,
                **ddp_kwargs,
            )
            self.text2latent = DistributedDataParallel(self.text2latent, **ddp_kwargs)
            self.mol2latent = DistributedDataParallel(self.mol2latent, **ddp_kwargs)

        ##### Freeze model parameters #####
        if args.representation_frozen:
            freeze_network(self.text_model)
            freeze_network(self.molecule_model)
            model_param_group = [
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        else:
            model_param_group = [
                {"params": self.text_model.parameters(), "lr": args.text_lr},
                {"params": self.molecule_model.parameters(), "lr": args.mol_lr},
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        
        self.optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        self.optimal_loss = 1e10        

    def gather_representations(self, tensor):
        if not self.distributed:
            return tensor
        return torch.cat(differentiable_all_gather(tensor), dim=0)

    def gather_without_grad(self, tensor):
        if not self.distributed:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)
    

    def get_text_repr_kd(self, text):
        """
        Get representation of molecular textual description with Language Models
        """
        text_tokens_ids, knowledge_masks, sentence_masks = prepare_text_tokens_kd(
            device=self.device,
            description=text,
            tokenizer=self.text_tokenizer,
            max_seq_len=self.args.max_seq_len,
            dynamic_padding=self.args.dynamic_padding,
        )
        
        # Get Representation of Original Sentence
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=knowledge_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)

        # Get Representation of Sentence including External Knowledge
        knowledge_output = self.ttext_model(input_ids=text_tokens_ids, attention_mask=sentence_masks)
        knowledge_repr = knowledge_output["pooler_output"]
        knowledge_repr = self.ttext2latent(knowledge_repr)
        
        return text_repr, knowledge_repr


    def update_teacher_model(self):
        # Update text model
        for s_params, t_params in zip(self.text_model.parameters(), self.ttext_model.parameters()):
            t_params.data = s_params.data
        # Update text2latent model
        for s_params, t_params in zip(self.text2latent.parameters(), self.ttext2latent.parameters()):
            t_params.data = s_params.data


    def train(self):

        for epoch in range(1, self.args.epochs + 1):

            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self.dataset.set_epoch(epoch)
            if self.rank == 0 and self.args.augmentation_strategy == "curriculum":
                print(
                    f"[Curriculum] epoch={epoch}, "
                    f"sampling_pool=top-{self.dataset.get_candidate_count()}"
                )
            if self.rank == 0 and self.args.augmentation_strategy == "stratified":
                high, mid, low = self.dataset.get_stratified_rank_groups()
                print(
                    "[Stratified] "
                    f"high=ranks {high[0] + 1}-{high[-1] + 1} (p={self.args.stratified_high_probability:.2f}), "
                    f"mid=ranks {mid[0] + 1}-{mid[-1] + 1} (p={self.args.stratified_mid_probability:.2f}), "
                    f"low=ranks {low[0] + 1}-{low[-1] + 1} (p={self.args.stratified_low_probability:.2f}), "
                    f"minimum_similarity={self.args.stratified_min_similarity:.2f}"
                )
            
            start_time = time.time()

            accum_loss, accum_distill_loss, accum_acc = 0, 0, 0
            completed_steps = 0

            for bc, samples in enumerate(tqdm(self.dataloader, disable=self.rank != 0)):

                if self.args.max_steps_per_epoch > 0 and bc >= self.args.max_steps_per_epoch:
                    break

                self.optimizer.zero_grad(set_to_none=True)

                description = samples[0]
                molecule = samples[1]
                rand_molecule = samples[2]
                aux_description = samples[3]
                
                defer_text_sync = self.distributed and self.args.alpha != 0
                with ExitStack() as sync_stack:
                    if defer_text_sync:
                        sync_stack.enter_context(self.text_model.no_sync())
                        sync_stack.enter_context(self.text2latent.no_sync())

                    with torch.cuda.amp.autocast(enabled=self.args.amp):
                        ##### Forward Pass: Language Model #####
                        description_repr = self.get_text_repr(description)

                        ##### Forward Pass: Molecule Model #####
                        molecule_repr = self.get_molecule_repr(rand_molecule)

                        all_description_repr = self.gather_representations(description_repr)
                        all_molecule_repr = self.gather_representations(molecule_repr)
                        original_fp = molecule.fp.to(self.device)
                        augmented_fp = rand_molecule.fp.to(self.device)
                        all_original_fp = self.gather_without_grad(original_fp)
                        all_augmented_fp = self.gather_without_grad(augmented_fp)

                        ##### Global in-batch Tanimoto soft targets #####
                        tanimoto_text_to_mol = calc_tanimoto(original_fp, all_augmented_fp)
                        tanimoto_mol_to_text = calc_tanimoto(augmented_fp, all_original_fp)

                        loss_01 = self.calc_S2P_Loss(
                            description_repr,
                            all_molecule_repr,
                            tanimoto_text_to_mol,
                        )
                        loss_02 = self.calc_S2P_Loss(
                            molecule_repr,
                            all_description_repr,
                            tanimoto_mol_to_text,
                        )
                        loss = (loss_01 + loss_02) / 2

                    # S2P uses the global batch. Text gradients are synchronized
                    # after the final ER microbatch; molecule gradients sync here.
                    self.scaler.scale(loss).backward()

                aux_batch_size = self.args.aux_batch_size or len(aux_description)
                distill_loss_value = 0.0
                aux_ranges = list(range(0, len(aux_description), aux_batch_size))
                for aux_chunk_number, aux_start in enumerate(aux_ranges):
                    aux_end = min(aux_start + aux_batch_size, len(aux_description))
                    aux_chunk = aux_description[aux_start:aux_end]
                    chunk_weight = len(aux_chunk) / len(aux_description)
                    is_final_aux_chunk = aux_chunk_number == len(aux_ranges) - 1
                    with ExitStack() as sync_stack:
                        if self.distributed and not is_final_aux_chunk:
                            sync_stack.enter_context(self.text_model.no_sync())
                            sync_stack.enter_context(self.text2latent.no_sync())
                        with torch.cuda.amp.autocast(enabled=self.args.amp):
                            aux_description_repr, aux_knowledge_repr = self.get_text_repr_kd(aux_chunk)
                            chunk_distill_loss = self.distill_loss(
                                aux_description_repr,
                                aux_knowledge_repr,
                            )
                        self.scaler.scale(
                            self.args.alpha * chunk_weight * chunk_distill_loss
                        ).backward()
                    distill_loss_value += chunk_weight * chunk_distill_loss.item()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.update_teacher_model()

                accum_loss += loss.item()
                accum_distill_loss += distill_loss_value
                completed_steps += 1

            if completed_steps == 0:
                raise RuntimeError("No training steps were completed in this epoch")

            temp_loss = accum_loss
            temp_distill_loss = accum_distill_loss
            if self.distributed:
                reduced_losses = torch.tensor(
                    [temp_loss, temp_distill_loss],
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.all_reduce(reduced_losses, op=dist.ReduceOp.SUM)
                temp_loss, temp_distill_loss = (
                    reduced_losses / self.world_size
                ).tolist()
            if not self.args.no_save and temp_loss < self.optimal_loss:
                self.optimal_loss = temp_loss
                self.save_model(epoch=epoch)
            if self.rank == 0:
                mean_s2p_loss = temp_loss / completed_steps
                mean_er_loss = temp_distill_loss / completed_steps
                weighted_er_loss = self.args.alpha * mean_er_loss
                mean_total_loss = mean_s2p_loss + weighted_er_loss
                print(
                    f"[Epoch {epoch}] CL Loss: {temp_loss:.5f}\t"
                    f"S2P Mean: {mean_s2p_loss:.5f}\t"
                    f"ER Mean: {mean_er_loss:.5f}\t"
                    f"Weighted ER Mean: {weighted_er_loss:.5f}\t"
                    f"Total Mean: {mean_total_loss:.5f}\t"
                    f"CL Acc: {accum_acc:.5f}\tTime: {time.time() - start_time:.5f}"
                )


if __name__ == "__main__":
    
    args, unknown = argument.parse_args()

    from models import AMOLE_Trainer
    model_trainer = AMOLE_Trainer(args)

    model_trainer.train()
