import torch
import torch.nn as nn
from packaging import version 
from typing import List

def conv_1x1_bn(
        in_channels: int,
        out_channels: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )

def conv_nxn_bn(
        in_channels: int,
        out_channels: int,
        kernel_size = 3,
        stride: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )

class FeedForward(nn.Module):
    def __init__(self,dim: int,hidden_dim: int,dropout: float = 0.,mode: str = 'simple') -> None:
        super(FeedForward,self).__init__()

        drop_layer = nn.Identity() if mode == 'simple' else nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            nn.SiLU(),
            drop_layer,
            nn.Linear(hidden_dim,dim),
            drop_layer
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int = 8,
            head_dim: int = 64,
            dropout: float = 0.,
            mode: str = 'simple',
    ) -> None:
       super(Attention,self).__init__()

       self.head_num = head_num
       inner_dim  = head_num * head_dim
       self.scale = head_dim ** -0.5
       self.drop_layer = nn.Identity() if mode == 'simple' else nn.Dropout(dropout)

       self.norm = nn.LayerNorm(dim)
       self.to_qkv = nn.Linear(dim,inner_dim * 3,bias=False)
       self.attend = nn.Softmax(dim=-1)

       self.to_out = nn.Sequential(
           nn.Linear(inner_dim,dim,bias=(mode == 'standard')),
           self.drop_layer
       )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        b,p,n,_ = x.shape

        x = self.norm(x)

        qkv: tuple[torch.Tensor] = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = [t.contiguous().view(b,p,n,self.head_num,-1).transpose(-2,-3) for t in qkv]

        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        dots = self.attend(dots)
        dots = self.drop_layer(dots)
        dots = torch.matmul(dots,v)

        dots = dots.transpose(-2,-3).contiguous().flatten(start_dim=-2)

        return self.to_out(dots)
    
class FlashAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int = 8,
            head_dim: int = 64,
            dropout: float = 0.,
            mode: str = 'simple'
    ) -> None:
        super(FlashAttention,self).__init__()

        assert not (version.parse(torch.__version__) < version.parse('2.0.0')),'in order to use flash attention, you must be using pytorch 2.0 or above'

        inner_dim = head_num * head_dim
        self.head_num = head_num
        self.backends = [
            nn.attention.SDPBackend.FLASH_ATTENTION,      # 优先级 1：极速引擎
            nn.attention.SDPBackend.EFFICIENT_ATTENTION,  # 优先级 2：显存优化引擎 (xFormers)
            nn.attention.SDPBackend.MATH                  # 优先级 3：传统矩阵乘法
         ]
        drop_layer = nn.Identity() if mode == 'simple' else nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim,inner_dim * 3,bias=False)
        self.to_out = nn.Sequential(
           nn.Linear(inner_dim,dim,bias=(mode == 'standard')),
           drop_layer
       )

    def flash_attn(self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,backends) -> torch.Tensor:
        with nn.attention.sdpa_kernel(backends):
            out = nn.functional.scaled_dot_product_attention(q,k,v)

        return out

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        b,p,n,_ = x.shape

        x = self.norm(x)
        qkv:tuple[torch.Tensor] = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = [t.contiguous().view(b * p,n,self.head_num,-1).transpose(-2,-3) for t in qkv]
        
        out = self.flash_attn(q,k,v,self.backends)
        out = self.to_out(out.transpose(-2,-3).contiguous().flatten(start_dim=-2))

        return out.contiguous().view(b,p,n,-1)

class Transformer(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            head_num: int,
            head_dim: int,
            depth: int,
            dropout: float = 0.,
            mode: str = 'simple',
            flash: bool = True
    ) -> None:
        super(Transformer,self).__init__()

        assert (mode == 'simple' or mode == 'standard'),'mode has only two options: \'simple\' and \'standard\''

        self.layers = nn.ModuleList()

        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim,head_num,head_dim,dropout,mode) if not flash else FlashAttention(dim,head_num,head_dim,dropout,mode),
                FeedForward(dim,hidden_dim,dropout,mode)
            ]))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        for attn,ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class MV2Block(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,stride: int = 1,expansion: float = 4):
        super(MV2Block,self).__init__()

        assert stride in [1,2]

        hidden_dim = int(in_channels * expansion)
        self.res_connect = stride == 1 and in_channels == out_channels

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,hidden_dim,3,stride,1,
                    groups=hidden_dim,bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),

                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),

                nn.Conv2d(
                    hidden_dim,hidden_dim,3,stride,1,
                    groups=hidden_dim,bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),

                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)

        if self.res_connect:
            return x + out
        else: 
            return out
        
class MobileViTBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            dim: int,
            kernel_size,
            hidden_dim: int,
            depth: int,
            patch_size: tuple[int,int],
            dropout: float = 0.,
            mode: str = 'simple',
            flash: bool = True
    ) -> None:
        super(MobileViTBlock,self).__init__()

        self.ph,self.pw = patch_size

        self.conv1 = conv_nxn_bn(in_channels,in_channels,kernel_size)
        self.conv2 = conv_1x1_bn(in_channels,dim)

        self.transformer = Transformer(dim,hidden_dim,4,8,depth,dropout,mode,flash)

        self.conv3 = conv_1x1_bn(dim,in_channels)
        self.conv4 = conv_nxn_bn(in_channels * 2,in_channels,kernel_size)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y) # [B(bf),dim,H,W]

        b,d,H,W = y.shape
        y = y.contiguous().view(b,d,(H // self.ph),self.ph,(W // self.pw),self.pw).permute(0,3,5,2,4,1).contiguous().view(b,self.ph * self.pw,-1,d)
        y = self.transformer(y)
        y = y.contiguous().view(b,self.ph,self.pw,(H // self.ph),(W // self.pw),d).permute(0,5,3,1,4,2).contiguous().view(b,d,H,W)

        y = self.conv3(y)
        y = torch.cat([x,y],dim=1)
        return self.conv4(y)

class TemporalHead(nn.Module):
    def __init__(self,in_channels: int,num_classes: int) -> None:
        super(TemporalHead,self).__init__()

        self.spatial_avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.temporal_net = nn.GRU(input_size=in_channels,hidden_size=in_channels,batch_first=True) # mark

        self.classifier = nn.Linear(in_channels,num_classes)

    def forward(self,x: torch.Tensor,B: int,F: int) -> torch.Tensor:
        x:torch.Tensor = self.spatial_avgpool(x).flatten(start_dim=1).view(B,F,-1) # x: [b*f,c]
        _,h_n = self.temporal_net(x)
        x = h_n.squeeze(0)
        return self.classifier(x)

class MobileViT(nn.Module):
    def __init__(
            self,
            image_size: tuple[int,int],
            dims: tuple[int,int,int],
            channels: List[int],
            num_classes: int,
            expansion: float = 4,
            kernel_size = 3,
            patch_size: tuple[int,int] = (2,2),
            depths: tuple[int,int,int] = (2,4,3),
            dropout: float = 0.
    ) -> None:
        super(MobileViT,self).__init__()

        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size

        assert ih % (32 * ph) == 0 and iw % (32 * pw) == 0, f"Image dimensions must be divisible by {32 * ph}."

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList()

        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[3], channels[4], 1, expansion)) 

        self.trunk = nn.ModuleList()
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[4], channels[5], 2, expansion),
            MobileViTBlock(
                in_channels=channels[5], dim=dims[0], kernel_size=kernel_size,
                hidden_dim=int(dims[0] * 2), depth=depths[0], dropout=dropout,
                patch_size=patch_size
            )
        ]))
        
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(
                in_channels=channels[6], dim=dims[1], kernel_size=kernel_size,
                hidden_dim=int(dims[1] * 4), depth=depths[1], dropout=dropout,
                patch_size=patch_size
            )
        ]))
        
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[6], channels[7], 2, expansion),
            MobileViTBlock(
                in_channels=channels[7], dim=dims[2], kernel_size=kernel_size,
                hidden_dim=int(dims[2] * 4), depth=depths[2], dropout=dropout,
                patch_size=patch_size
            )
        ]))

        self.conv_final_1x1 = conv_1x1_bn(channels[7],last_dim)
        self.to_logits = TemporalHead(last_dim,num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"Expected 5D video tensor [B, F, C, H, W], but got {x.dim()}D."

        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)

        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        x = self.conv_final_1x1(x)
        return self.to_logits(x,B,F)

        

        



        

        


