import torch
import os

local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
x = torch.randn(4,4, device=torch.cuda.current_device())
