import torch
from torch import nn    


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # add small value to prevent division by zero
        self.weight = nn.Parameter(torch.ones(d_model)) # add leqrnable weight parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight 
    
