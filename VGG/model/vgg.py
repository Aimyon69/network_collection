import torch
import torch.nn as nn
from typing import Union,List,Dict,Any,cast

_cfgs:Dict[str,List[Union[str,int]]] = {
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5
    ) -> None:
        """
        VGG Model Main Class.
        
        Args:
            features (nn.Module): The convolutional feature extraction part.
            num_classes (int): Number of classes for classification. Default: 1000 (ImageNet).
            init_weights (bool): Whether to initialize weights using Kaiming Init.
            dropout (float): Dropout rate.
        """
        super(VGG,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._initialize_weights()
        
    def forward(self,x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

def make_layers(cfg: List[Union[str,int]],batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            v = cast(int,v)
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers.extend([conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d,nn.ReLU(inplace=True)])
            in_channels = v

    return nn.Sequential(*layers)
    
def vgg11(pretrained: bool = False,progress: bool = True,**kwargs: Any) -> VGG:
    if pretrained:
        pass

    model = VGG(make_layers(_cfgs['vgg11'],batch_norm=True),**kwargs)
    return model

def vgg13(pretrained: bool = False,progress: bool = True,**kwargs: Any) -> VGG:
    if pretrained:
        pass

    model = VGG(make_layers(_cfgs['vgg13'],batch_norm=True),**kwargs)
    return model

def vgg16(pretrained: bool = False,progress: bool = True,**kwargs: Any) -> VGG:
    if pretrained:
        pass

    model = VGG(make_layers(_cfgs['vgg16'],batch_norm=True),**kwargs)
    return model

def vgg19(pretrained: bool = False,progress: bool = True,**kwargs: Any) -> VGG:
    if pretrained:
        pass

    model = VGG(make_layers(_cfgs['vgg19'],batch_norm=True),**kwargs)
    return model