import torch
import torch.nn as nn
from typing import Any,List
import math
import torch.nn.functional as F

_Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expansion: int
    ) -> None:
        super(Bottleneck,self).__init__()

        self.connect = in_channels == out_channels and stride == 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,out_channels=in_channels * expansion,kernel_size=1,
                stride=1,padding=0,bias=False
            ),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),

            nn.Conv2d(
                in_channels= in_channels * expansion,out_channels=in_channels * expansion,
                groups=in_channels * expansion,kernel_size=3,stride=stride,padding=1,bias=False
            ),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),

            nn.Conv2d(
                in_channels= in_channels * expansion,out_channels=out_channels,
                kernel_size=1,stride=1,padding=0,bias=False
            ),
            nn.BatchNorm2d(out_channels)
            # no PReLU
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Any,
            stride: int,
            padding: int,
            depthwise: bool = False,
            linear: bool = False 
    ) -> None:
        super(ConvBlock,self).__init__()

        self.linear = linear

        if depthwise:
            self.conv = nn.Conv2d(
                in_channels=in_channels,out_channels=out_channels,
                groups=in_channels,kernel_size=kernel_size,
                stride=stride,padding=padding,bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,out_channels=out_channels,
                kernel_size=kernel_size,stride=stride,
                padding=padding,bias=False
            )

        self.bn = nn.BatchNorm2d(out_channels)

        if not linear:
            self.prelu = nn.PReLU(out_channels)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if not self.linear:
            x = self.prelu(x)

        return x
    
class MobileFaceNet(nn.Module):
    def __init__(self,bottleneck_setting: List[tuple[int,int,int,int]] = _Mobilefacenet_bottleneck_setting) -> None:
        super(MobileFaceNet,self).__init__()

        self.conv3x3 = ConvBlock(
            in_channels=3,out_channels=64,
            kernel_size=3,stride=2,padding=1
        )

        self.dwc3x3 = ConvBlock(
            in_channels=64,out_channels=64,
            kernel_size=3,stride=1,padding=1,depthwise=True
        )

        self.bottleneck = self._make_layers(bottleneck_setting)

        self.conv1x1 = ConvBlock(
            in_channels=128,out_channels=512,
            kernel_size=1,stride=1,padding=0
        )

        self.GDC = ConvBlock(in_channels=512,out_channels=512,kernel_size=(7,6),stride=1,padding=0,depthwise=True,linear=True) # Linear ?

        self.linear1x1 = ConvBlock(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0,linear=True)

        self._init_weights()

    def _make_layers(self,bottleneck_setting: List[tuple[int,int,int,int]]) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 64

        for t,c,n,s in bottleneck_setting:
            for i in range(n):
                if i == 0:
                    layers.append(Bottleneck(in_channels=in_channels,out_channels=c,stride=s,expansion=t))
                else:
                    layers.append(Bottleneck(in_channels=c,out_channels=c,stride=1,expansion=t))
            in_channels = c

        return nn.Sequential(*layers) 

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.conv3x3(x)
        x = self.dwc3x3(x)
        x = self.bottleneck(x)
        x = self.conv1x1(x)
        x = self.GDC(x)
        x = self.linear1x1(x)
        x = x.view(x.shape[0],-1)

        return x
    
class ArcMarginProduct(nn.Module): #TODO
    def __init__(self,in_channels: int,out_channels: int,s: float,m: float,easy_margin: bool = False) -> None:
        super(ArcMarginProduct,self).__init__()

        self.weights = nn.Parameter(torch.Tensor(out_channels,in_channels))
        nn.init.xavier_uniform_(self.weights)
        self.easy_margin = easy_margin
        self.s = s

        self.sin_m = math.sin(m)
        self.cos_m = math.cos(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self,x: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x),F.normalize(self.weights))
        
        target_logit = torch.gather(cosine, 1, label.view(-1, 1))

        sine = torch.sqrt(1 - torch.pow(target_logit,2))
        phi = target_logit * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            target_logit = torch.where(target_logit > 0,phi,target_logit)
        else:
            target_logit = torch.where(target_logit > self.th,phi,target_logit - self.mm)
        onehot_mask = torch.zeros_like(cosine)
        onehot_mask.scatter_(1, label.view(-1, 1), 1)
        output = cosine * (1 - onehot_mask) + target_logit * onehot_mask

        return output * self.s


