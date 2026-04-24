import torch
import torch.nn as nn

""" some of annotations stand for the standard ViT implementation """

def posemb_sincos_2d(
        h: int,
        w: int,
        dim: int,
        temperature: int = 10000,
        dtype = torch.float32
) -> torch.Tensor:
    assert (dim % 4 == 0),"feature dimension must be multiple of 4 for sincos emb"

    y,x = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij') # y,x: [h,w]

    omega = torch.arange(dim // 4) / (dim // 4 - 1) # omega: [dim // 4]
    omega = 1.0 / (temperature ** omega) # omega: [dim // 4]

    y = y.flatten(start_dim=0)[:,None] * omega[None,:] # y: [h * w,dim // 4]
    x = x.flatten(start_dim=0)[:,None] * omega[None,:] # x: [[h * w,dim // 4]]

    return torch.cat([x.sin(),x.cos(),y.sin(),y.cos()],dim=-1).type(dtype) # return: [h * w,dim]

class FeedForward(nn.Module):
    def __init__(self,dim: int,hidden_dim: int) -> None:
        super(FeedForward,self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            # nn.Dropout(dropout)
            nn.GELU(),
            nn.Linear(hidden_dim,dim)
            # nn.Dropout(dropout)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_head: int = 8,
            head_dim: int = 64
    ) -> None:
        super(Attention,self).__init__()

        inner_dim = num_head * head_dim
        self.num_head = num_head
        self.scale = head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim,inner_dim * 3,bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Linear(inner_dim,dim,bias=False)
        """ self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() """

    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        b,n,_ = x.shape

        x = self.norm(x)

        qkv: tuple[torch.Tensor] = self.to_qkv(x).chunk(3,dim=-1) # Tuple[3 * torch.Tensor[b,n,inner_dim]]
        q,k,v = [t.contiguous().view(b,n,self.num_head,-1).transpose(-2,-3) for t in qkv] # q,k,v: [b,h,n,head_dim]
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale # dots: [b,h,n,n]
        dots = self.attend(dots)
        dots = torch.matmul(dots,v) # dots: [b,h,n,head_dim]

        dots = dots.transpose(-2,-3).flatten(start_dim=-2)

        return self.to_out(dots)
    
class Transformer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_head: int,
            head_dim: int,
            hidden_dim: int,
            depth: int
    ) -> None:
        super(Transformer,self).__init__()
        
        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_head, head_dim),
                FeedForward(dim, hidden_dim)
            ]))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        for attn,ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        return self.norm(x)
    
class SimpleViT(nn.Module):
    def __init__(
            self,
            image_size: tuple[int,int],
            patch_size: tuple[int,int],
            num_classes: int,
            dim: int,
            num_head: int,
            depth: int,
            mlp_dim: int,
            head_dim: int = 64,
            image_channels: int = 3,
    ) -> None:
        super(SimpleViT,self).__init__()

        assert (image_size[0] % patch_size[0] == 0) and (image_size[1] % patch_size[1] == 0),'Image dimensions must be divisible by the patch size.'

        self.proj = nn.Conv2d(in_channels=image_channels,out_channels=dim,kernel_size=patch_size,stride=patch_size)

        self.pos_embedding = posemb_sincos_2d(
            h = image_size[0] // patch_size[0],
            w = image_size[1] // patch_size[1],
            dim = dim
        )
        self.register_buffer('pos_embedding',self.pos_embedding,persistent=False)

        self.encoder = Transformer(dim,num_head,head_dim,mlp_dim,depth)

        self.to_latent = nn.Identity()

        self.mlp = nn.Linear(dim,num_classes)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # x: [b,dim,image_size[0] // patch_size[0],image_size[1] // patch_size[1]]
        x = x.flatten(start_dim=-2).transpose(-1,-2) # x: [b,n,dim]

        # x = torch.cat([cls_embedding,...])...
        x = x + self.pos_embedding # implicit up-dimensional
        # x = self.dropout(x)
        x = self.encoder(x)

        x = torch.mean(x,dim=1) # or cls pool

        x = self.to_latent(x)

        return self.mlp(x) 




        
        
