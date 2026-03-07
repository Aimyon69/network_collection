import torch
import torch.nn as nn
from .config import cfgs
from typing import List,Union,Dict
import torchvision.models as models
from collections import OrderedDict

class vgg_deconv(nn.Module):
    def __init__(self,model_name: str) -> None:
        super(vgg_deconv,self).__init__()

        self.features = make_layers(model_name)

        self.conv2deconv_indices,self.unpool2pool_indices = self.init_indices()

        self.init_pretrained_weights(model_name)

    def  init_pretrained_weights(self,model_name: str) -> None:
        try:
            pretrained_model_func = getattr(models,model_name)
            pretrained_model = pretrained_model_func(pretrained=True)            
        except Exception as e:
            print(e)
            return
        
        for idx,layer in enumerate(pretrained_model.features):
            if isinstance(layer,nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                #self.features[self.conv2deconv_indices[idx]].bias.data = layer.bias.data

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


def search_channel(cfg: List[Union[str,int]],start_index: int):
    if start_index < 0 or start_index >= len(cfg):
        raise IndexError('index out of range')
    
    start_index -= 1
    while  start_index >= 0:
        if cfg[start_index] != 'M':
            break
        start_index -= 1

    return start_index

def make_layers(model_name: str) -> nn.Sequential:
    try:
        modules = cfgs[model_name]
    except Exception as e:
        print(e)
        return
    
    layers: List[nn.Module] = []
    
    for idx in range(len(modules)-1,-1,-1):
        if modules[idx] == 'M':
            layers.append(nn.MaxUnpool2d(kernel_size=2,stride=2))
        else:
            index = search_channel(modules,idx)
            if index == -1:
                layers.extend([
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=modules[idx],out_channels=3,kernel_size=3,padding=1)
                ])
            else:
                layers.extend([
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=modules[idx],out_channels=modules[index],kernel_size=2,padding=1)
                ])
    
    return nn.Sequential(*layers)
