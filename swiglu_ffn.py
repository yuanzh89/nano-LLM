import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.w1(x)
        o2 = self.w2(x)
        o2 = F.silu(o2)
        o3 = self.w3(o1 * o2)
        return o3