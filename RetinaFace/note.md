# RetinaFace

本文对RetinaFace框架做一个总结分析，涉及**模型架构分析，搭建，训练策略，前后处理逻辑**。需要强调的是：本次实现框架是基于**计算资源紧张的边缘设备端**部署场景，摒弃了原架构中**DCN（Deformable Convolution Network--可变形卷积）以及 Dense Regression Branch**，得到的轻量级简化框架以达到实时性与精准性之间的平衡。

---

## 基本子模块（Basic Sub-Module）

```python
def conv_bn(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(in_channels: int,out_channels: int,stride :int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
        nn.BatchNorm2d(out_channels)
    )

def conv1x1(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=stride,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )

def conv_dw(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,stride=stride,bias=False,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )
```

大
