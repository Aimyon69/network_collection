import torch
import torch.nn.functional as F

def posemb_sincos_3d(
        f: int, 
        h: int, 
        w: int, 
        dim: int,
        temperature: int = 10000,
        dtype = torch.float32
) -> torch.Tensor:    
    z,y,x = torch.meshgrid(
        torch.arange(f),
        torch.arange(h),
        torch.arange(w),
        indexing='ij'
    ) # z,y,x: [f,h,w]

    omega = torch.arange(dim // 6) / (dim // 6 - 1)
    omega = 1.0 / temperature ** omega

    z = z.flatten(start_dim=0)[:,None] * omega[None,:] # z: [f * h * w,dim // 6]
    y = y.flatten(start_dim=0)[:,None] * omega[None,:]
    x = x.flatten(start_dim=0)[:,None] * omega[None,:]

    pe = torch.cat([x.sin(),x.cos(),y.sin(),y.cos(),z.sin(),z.cos()],dim=-1)

    pe = F.pad(pe,(0,dim - dim // 6 * 6))  # pad if feature dimension not cleanly divisible by 6
    """same as the code as follows:
            zeros = torch.zeros(f * h * w,dim - dim // 6 * 6)
            pe = torch.cat([pe,zeros],dim=-1)"""
    
    return pe.type(dtype)

