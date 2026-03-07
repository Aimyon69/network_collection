import torch
import torch.nn as nn
from typing import List,Dict
from collections import OrderedDict

class anynet_deconv(nn.Module):
    def __init__(self,conv_model: nn.Module) -> None:
        super(anynet_deconv,self).__init__()

        self.features = self.init_features(conv_model)

        self.conv2deconv_indices,self.unpool2pool_indices = self.init_indices()

    def init_features(self,conv_model: nn.Module) -> nn.Sequential:
        layers: List[nn.Module] = []

        modules = conv_model._modules.get('features')
        if modules is None:
            raise ValueError('conv_model doesnt have standard features')
        
        for module in modules:
            if isinstance(module,nn.MaxPool2d):
                MUP = nn.MaxUnpool2d(kernel_size=module.kernel_size,stride=module.stride)
                layers.append(MUP)
            elif isinstance(module,nn.Conv2d):
                deconv = nn.ConvTranspose2d(
                    in_channels = module.out_channels,
                    out_channels= module.in_channels,
                    kernel_size = module.kernel_size,
                    padding = module.padding,
                    stride = module.stride
                )
                deconv.weight.data = module.weight.data
                layers.append(deconv)
            elif isinstance(module,nn.ReLU):
                relu = nn.ReLU(inplace = module.inplace)
                layers.append(relu)

        layers.reverse()
        return nn.Sequential(*layers)         
    
    def init_indices(self) -> List[dict]:
        conv2deconv_indices: Dict[int,int] = {}
        unpool2pool_indices: Dict[int,int] = {}
        num_layers = len(self.features)

        for idx,layer in enumerate(self.features):
            if isinstance(layer,nn.MaxUnpool2d):
                unpool2pool_indices[idx] = num_layers - 1 -idx
            elif isinstance(layer,nn.ConvTranspose2d):
                conv2deconv_indices[num_layers - 1 -idx] = idx

        return [conv2deconv_indices,unpool2pool_indices]
    
    def forward(
            self,
            x: torch.Tensor,
            layer: int,
            pool_locs: OrderedDict[int,torch.Tensor] 
    ) ->torch.Tensor:
        if layer in self.conv2deconv_indices:
            start_index = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv')

        for idx in range(start_index,len(self.features)):
            if isinstance(self.features[idx],nn.MaxUnpool2d):
                x = self.features[idx](
                    x,pool_locs[self.unpool2pool_indices[idx]]
                )
            else:
                x = self.features[idx](x)
        return x