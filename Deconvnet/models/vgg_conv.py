import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from .config import cfgs
from typing import List

class vgg_conv(nn.Module):
    def __init__(
            self,
            model_name: str,
            dropout_ratio: float = 0.5,
            num_classes: int = 1000
    ) -> None:
        super(vgg_conv,self).__init__()

        self.features = make_layers(model_name)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(4096,num_classes)
        )

        self.feature_maps: OrderedDict[int,torch.Tensor] = OrderedDict()
        self.pool_locs: OrderedDict[int,torch.Tensor] = OrderedDict()

        self.init_pretrained_weights(model_name) 

    def init_pretrained_weights(self,model_name: str) -> None:
        try:
            pretrained_model_func = getattr(models,model_name)
            pretrained_model = pretrained_model_func(pretrained=True)
        except Exception as e:
            print(e)
            return
        
        for idx,layer in enumerate(pretrained_model.features):
            if isinstance(layer,nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data

        for idx,layer in enumerate(pretrained_model.classifier):
            if isinstance(layer,nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data 

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        for idx,layer in enumerate(self.features):
            if isinstance(layer,nn.MaxPool2d):
                x,location = layer(x)
                self.pool_locs[idx] = location
            else:
                x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def make_layers(model_name: str) -> nn.Sequential:
    try:
        modules = cfgs[model_name]
    except Exception as e:
        print(e)
        return 

    layers: List[nn.Module] = []
    in_channels = 3

    for module in modules:
        if module == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True))
        else:
            layers.extend([nn.Conv2d(in_channels=in_channels,out_channels=module,kernel_size=3,padding=1),
                           nn.ReLU(inplace=True)])
            in_channels = module
        
    return nn.Sequential(*layers)
