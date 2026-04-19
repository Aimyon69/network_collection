import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.
    ) -> None:
        super(FeedForward,self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim), # mark
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_head: int,
            head_dim: int,
            dropout: float = 0.
    ) -> None:
        super(Attention,self).__init__()

        project_out = not (num_head == 1 and head_dim == dim)
        inner_dim = num_head * head_dim
        self.scale = head_dim ** -0.5
        self.num_head = num_head

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim,inner_dim * 3)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x) # mark

        qkv: torch.Tensor = self.to_qkv(x) # qkv: [b,n,inner_dim * 3]
        qkv = qkv.chunk(3,dim=-1) # qkv: Tuple(3 * Tensor[b,n,inner_dim])

        b,n,_ = x.shape
        q, k, v = [t.contiguous().view(b, n, self.num_head, -1).transpose(-2, -3) for t in qkv]

        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale

        dots = self.attend(dots)
        dots = self.dropout(dots)

        output = torch.matmul(dots,v) # output: [b,h,n,d]
        
        output = output.transpose(-2,-3).contiguous().view(b,n,-1)

        return self.to_out(output)

class Transformer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_head: int,
            head_dim: int,
            depth: int,
            hidden_dim: int,
            dropout: float = 0.
    ) -> None:
        super(Transformer,self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim,num_head,head_dim,dropout),
                FeedForward(dim,hidden_dim,dropout)
            ]))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: each Attention and FeedForward module has internal LayerNorm
        # Residual connections are implemented at the Transformer block level
        for at,ff in self.layers:
            x = x + at(x)
            x = x + ff(x)

        return self.norm(x)

class ViT(nn.Module):
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
            pool: str = "cls",
            dropout: float = 0.,
            emb_dropout: float = 0.
            
    ) -> None:
        super(ViT,self).__init__()

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'

        self.proj = nn.Conv2d(in_channels=image_channels,out_channels=dim,kernel_size=patch_size,stride=patch_size)

        assert pool == "cls" or pool == "mean",'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == "cls" else 0
        self.cls_embedding = nn.Parameter(torch.randn(1,num_cls_tokens,dim)) # cls_embedding: [b(1),1,dim]; empty when pool="mean" (num_cls_tokens=0)

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches + num_cls_tokens,dim)) # mark
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Transformer(dim,num_head,head_dim,depth,mlp_dim,dropout)

        self.pool = pool
        
        self.to_latent = nn.Identity() # face-future coding

        self.mlp = nn.Linear(dim,num_classes)
        
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # raw image to patches
        x = self.proj(x) # x: [b,dim,image_size[0] // patch_size[0],image_size[1] // patch_size[1]]
        x = torch.flatten(x,start_dim=-2) # x: [b,dim,n]
        x = x.transpose(-1,-2) # x: [b,n,dim]
        # cls embedding and pos embedding
        b = x.shape[0]
        x = torch.cat([self.cls_embedding.expand(b,-1,-1),x],dim=1) # x: [b,n(patch) + 1(cls),dim]
        x = x + self.pos_embedding.expand(b,-1,-1)
        x = self.dropout(x)
        # encode
        x = self.encoder(x) # x: [b,n,dim]

        x = torch.mean(x,dim=1) if self.pool == "mean" else x[:,0] # x: [b,dim]

        x = self.to_latent(x)

        return self.mlp(x)
        