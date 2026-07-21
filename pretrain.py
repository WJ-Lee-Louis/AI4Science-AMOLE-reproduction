# Set-up the environment variable to ignore warnings
import warnings
warnings.filterwarnings('ignore')

from utils import argument
import random
import os

import numpy as np
import torch
import torch.distributed as dist

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.set_num_threads(2)
os.environ['OMP_NUM_THREADS'] = "2"


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain():

    args, unknown = argument.parse_args()
    if unknown:
        raise ValueError(f"Unknown command-line arguments: {unknown}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        args.device = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
    seed_everything(args.seed)

    try:
        if args.model == 'AMOLE':
            from models import AMOLE_Trainer
            model_trainer = AMOLE_Trainer(args)

        model_trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":

    pretrain()
