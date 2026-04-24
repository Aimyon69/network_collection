import torch
import torch.nn as nn
from packaging import version
from collections import namedtuple

# cuda_backends = [
#             nn.attention.SDPBackend.FLASH_ATTENTION,      # 优先级 1：极速引擎
#             nn.attention.SDPBackend.EFFICIENT_ATTENTION,  # 优先级 2：显存优化引擎 (xFormers)
#             nn.attention.SDPBackend.MATH                  # 优先级 3：传统矩阵乘法
#         ]

class Attend(nn.Module):
    def __init__(
            self,
            use_flash: bool,
            backends: list[nn.attention.SDPBackend]
    ) -> None:
        super(Attend,self).__init__()

        assert not (version.parse(torch.__version__) < version.parse('2.0.0')),'in order to use flash attention, you must be using pytorch 2.0 or above'
        
        self.use_flash = use_flash

        self.backends = backends

    def flash_attn(self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
        with nn.attention.sdpa_kernel(self.backends):
            out = nn.functional.scaled_dot_product_attention(q,k,v)

        return out

    def forward(self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q,k,v)
        
        sim: torch.Tensor = torch.einsum("b h i d, b h j d -> b h i j ",q,k) * scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d",attn,v)

        return out