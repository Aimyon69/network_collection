# VGG Notes

---

这是对VGG Network的总结笔记，总结按照论文排版顺序。

[TOC]


---

## 去均值化（Mean Subtraction）or 零中心化（Zero-Centering）

>During training, the input to our ConvNets is a fixed-size 224 × 224 RGB image. The only pre-processing we do is subtracting the mean RGB value, computed on the training set, from each pixel.

`在训练中，我们网络的输入图像大小固定为224x224.对输入图像的每一个像素都减去我们在训练数据集计算出来的对应通道的RGB像素值均值，这是我们唯一做的预处理。`

在之前的学习过程中已经提前遇到了这个操作（RetinaFace），说明这对于CNN网络来说是必须且有益的。那么这个操作给网络带来了什么？

- ***零中心化：***将数据分布在原点周围（数据均值在0附近），使得特征的数值波动围绕原点，避免均值偏移导致模型对数据均值的依赖，导致泛化性下降，导致无用学习
- ***避免权重更新方向一致：***模型权重的更新公式是$ \sum_{} W_iX_i+b$，如果对于原始图像像素值输入，那么一层网络中的所有权重的更新方向完全由那一层的梯度正负决定，换句话说，就是同一层的所有权重的更新方向一致。这对于模型的收敛是致命的，因为显然模型收敛到最佳，一定是某些权重系数增大（关键特征），一些权重减小（无关特征，噪声），更新权重方向固定阻碍了模型收敛，导致模型权重更新呈现一会大幅增大，一会大幅减小，这就是**Zig-Zag（锯齿）**。
- ***有利于分类直线切分：***数据包围着原点。你的分类直线直接穿过原点（或者在原点附近），只需要轻轻旋转角度就能很好地切分数据。这使得参数优化更加**鲁棒（Robust）**数据都在第一象限，离原点很远。你的分类直线必须非常小心地调整角度和截距，微小的角度变化都会导致直线剧烈摆动，很容易错过数据簇。

这一预处理操作会强制要求在**test阶段**也要对输入图像进行减去训练数据集RGB通道均值，由于大部分的模型都是在ImageNet数据集上训练的，这个均值就固定为：**R：123.68，G: 116.78，B: 103.94**。对于其他数据集训练的模型这三个值具有差异，根据具体的细节确定。

---

## 网络配置（ConvNet Configurations）

![image-20260203211410853](.\image-20260203211410853.png)

---

## 小卷积核代替大卷积核

> Rather than using relatively large receptive fields in the first conv. layers (e.g. 11 × 11 with stride 4 in (Krizhevsky et al., 2012), or 7 × 7 with stride 2 in (Zeiler & Fergus,2013; Sermanet et al., 2014)), we use very small 3 × 3 receptive fields throughout the whole net,which are convolved with the input at every pixel (with stride 1). It is easy to see that a stack of two 3 × 3 conv. layers (without spatial pooling in between) has an effective receptive field of 5 × 5; three such layers have a 7 × 7 effective receptive field.

`（总结性翻译）我们用3x3大小的卷积核代替了在此之前模型常用的5x5，7x7大小的卷积核。`

>The incorporation of 1 × 1 conv. layers (configuration C, Table 1) is a way to increase the non-linearity of the decision function without affecting the receptive fields of the conv. layers. Even though in our case the 1 × 1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function.

`（总结性翻译）我们利用了1x1卷积核来增加网络的非线性，即使1x1卷积本质上是对同维度空间的线性投影且不影响感受野大小，但是激活函数引入了额外的非线性`

在上述的两个论文叙述可以看出，在VGG中，模型摒弃了大卷积核的使用。下面就来阐述下两者的对比，首先给出深层卷积感受野计算公式：
$$
RF_n = RF_{n-1} + (k_n - 1) \cdot S_{n-1}
$$
其中RF为感受野大小，n代表网络深度，k代表卷积核大小，S是前面n-1层的累积步长。

- ***参数量减少：***由以上的公式易得：两个3x3卷积层堆叠等效于5x5卷积，三个3x3卷积层堆叠等效为7x7卷积。拿第二个等效举例子，设输入输出通道数是C，三个3x3卷积的参数为27C^2^，对于一个7x7卷积来说，参数量为49C^2^。明显的参数量差距，显然前面的计算速度和效率更高。
- ***更多的非线性引入：***小卷积核堆叠代替大卷积核，会引入两个额外的ReLU激活函数（对于7x7来说），使得网络的非线性拟合能力增强，能够提取更加复杂的特征，模型性能增强。

1x1卷积层对网络的感受野没有影响，但是也是引入了更多的非线性，使得模型的性能更强。

---

## 训练策略（Training Strategy）

>The ConvNet training procedure generally follows Krizhevsky et al. (2012) (except for sampling the input crops from multi-scale training images, as explained later). Namely, the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent
>(based on back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 256,momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5 · 10−4) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving. 

可以从上面的描述中总结出四个训练策略：

- ***动量（Momentum）：***动量的引入会引起模型参数更新会依赖之前参数更新方向，这是一种对**之前经验的信任**，这种信任会让模型不会在局部最优值附近震荡，让其跳出局部最优值，朝着全局最优值收敛。即使不存在局部最优的情况，动量也会使得模型在同一梯度下降方向收敛得更快。

- ***权重衰减（Weight Decay）：***$$Loss_{total} = Loss_{data} + \lambda \sum w^2$$权重衰减的本质就是在损失函数中添加一项权重平方和的惩罚项，其中的$\lambda$就是惩罚系数，论文中提到其值为5e^-4^，在参数更新求偏导的时候，会让权重去减去一项系数，具体的公式为：$w_{new} = w_{old} - \text{学习率} \times (\text{梯度} + \lambda w_{old})$​，使得新权重的绝对值变小。让权重的绝对值变小的目的是为了减小因为网络输入值的微小变化，但由于网络层的权重值过大，导致输出值波动较大，模型难以收敛且精度低的问题。且大权重的卷积核的存在会使得小权重卷积核的“失活”，让网络的推理结果由大卷积核主导，推理过度依赖少数特征（主导特征），网络失去特征多样性。

- ***神经元随机失活（Dropout）：***这是一种防止模型过拟合的很好的正则化方法，而且这种方法会使模型训练的时候参数量减小，计算更快，模型收敛更快。本质就是在**正向传播（Forward Propagation）**和**反向传播（Backward Propagation）**过程中将某些神经元暂时“移除”（逻辑上的移除），使其不参与正向传播的输出值计算和反向传播的梯度计算和参数更新。神经元失活的比例由dropout ratio决定，论文中设置的是0.5，将此方法运用在了**Fully Connected Layers（全连接层）**。这个方法使得模型的输出值不过度依赖于特定的神经元，模型能学到更加普遍的特征，避免了过拟合。

- ***动态学习率（Dynamic Learning Rate）：***论文中提到，学习率一共下降三次，在验证集精度不再改善的时候将学习率变为原来的1/10。这样会使得模型更加收敛于全局最优。

---

## 预初始化（Pre-Initialization）

> In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs). We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv. filter sizes; (b) pre-initialisation of certain layers.

首先论文指出了一个非常有趣但是反直觉的现象：模型深度更深，参数更多的VGG训练收敛所需的epochs比模型深度浅，参数少的AlexNet网络收敛所需epochs少。他们推断现象原因为**（a）更深的网络和更小的卷积核引入了隐式正则化。（b）特定网络层的预初始化。**对于（a）换种话来说：作者认为，使用 3x3**小卷积核**多层堆叠，不仅减少了单一层的参数，还在保持感受野的同时增加了非线性（因为层数多了，ReLU 也多了）。这种结构本身就迫使网络学习更有意义的特征，起到了正则化的作用。

>The initialisation of the network weights is important, since bad initialisation can stall learning due to the instability of gradient in deep nets. To circumvent this problem, we began with training the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when training deeper architectures, we initialised the first four convolutional layers and the last three fully-connected layers with the layers of net A (the intermediate layers were initialised randomly). We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean and 10−2 variance. The biases were initialised with zero. It is worth noting that after the paper submission we found that it is possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).

这一段是对预初始化的具体阐述，首先提出网络权重初始化是非常重要的，因为不好的权重初始化会导致深层网络的梯度不稳定使得学习停滞。这个揭示了一个深层网络的核心痛点：网络越深，模型越难收敛，因为深层网络带来的梯度爆炸和梯度消失的问题。所以对于VGG时代，VGG19的训练是一件困难的事情。所以为了绕过这个困难，论文先训练了VGG11，其浅得足以用随机初始化训练。然后将VGG11的前四个卷积层和最后的三个全连接层权重利用于更深的VGG网络初始化，中间层利用随机初始（均值为0，方差为10^-2^的正态分布中采样权重值，bias置0），且没有下降预初始化的层的学习率，允许预初始化层的参数随着训练更新**（Fine-tuning）**。

而为什么选择前四个卷积层和后面的三个全连接层做后面深层网络的初始化呢？因为前四个卷积层提取的是图像中通用的纹理，边缘，颜色特征，这一部分是通用的。后面三层的全连接起到语义分类的作用，也是通用的。深层网络的中间层则是将低级特征整合成高级特征，由于浅层网络不具有这样的能力，故采用随机初始化。

最后的一句说：` It is worth noting that after the paper submission we found that it is possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).（值得注意的是，在论文提交后我们发现，其实可以不需要这种预训练步骤，而是直接使用 Glorot & Bengio (2010) 提出的随机初始化过程（也就是 Xavier 初始化）来初始化权重。）`这种方法显然比先训练浅层网络更加高效方便，这也是现在的主流做法：现在的深度学习框架（PyTorch/TensorFlow）默认都已经使用了 **Xavier (Glorot)** 或者 **Kaiming (He)** 初始化。

---

## 数据增强（Data Augmentation）

这一小节阐述VGG采用的一系列数据增强方法。

### 随机裁剪（Randomly Crop）

>To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration).

在缩放后的训练集图像随机裁剪224大小的输入图像，这一步使得模型输入图像不再是固定的目标物体全貌图像居中，由于裁剪随机性，可能是目标物体的部分图像，位置也具有随机性，强迫模型学习目标物体的局部特征，而不过度依赖位置居中性，物体整体性。提升了鲁棒性。

### 水平翻转（Horizontal Flipping）和颜色抖动（RGB Colour Shift）

>To further augment the training set, the crops underwent random horizontal flipping and random RGB colour shift (Krizhevsky et al., 2012).

`为了进一步扩充训练集（数据增强），这些裁剪出来的图像还经过了随机水平翻转和随机 RGB 颜色偏移（参考 AlexNet 的做法）。`

这两个都是常见的数据增强方法，拓展了训练数据集的数量，同时提高了模型的鲁棒性。后者通过改变光照颜色，提高了模型对光照变化的鲁棒性。

### 缩放训练集图像（Rescaled Training Images）

先引入必要概念：

>Let S be the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped (we also refer to S as the training scale). While the crop size is fixed to 224 × 224, in principle S can take on any value not less than 224: for S = 224 the crop will capture whole-image statistics, completely spanning the smallest side of a training image; for S ≫ 224 the crop will correspond to a small part of the image, containing a small object or an object part.

`设S为经过等比缩放（isotropically-rescaled）后的训练图像的最短边长度，卷积网络的输入就是从这个缩放后的图像中裁剪出来的（我们也把S称为训练尺度）。虽然网络输入的裁剪尺寸固定为224x224，但原则上S可以是任何不小于224的数值，如果S=224，裁剪出的图像就会包含整张图的统计特征（因为正好覆盖了图片的最短边）；如果S远大于224，裁剪出来的就只是原图的一小部分，可能只包含一个小物体或者是物体的一部分。`

#### 单尺度训练（Single-Scale Training）

>The first is to fix S, which corresponds to single-scale training (note that image content within the sampled crops can still represent multi-scale image statistics). In our experiments, we evaluated models trained at two fixed scales: S = 256 (which has been widely used in the prior art (Krizhevsky et al., 2012; Zeiler & Fergus, 2013;Sermanet et al., 2014)) and S = 384. Given a ConvNet configuration, we first trained the network using S = 256. To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 10−3.

对于上面的具体叙述不做详细的说明，但值得注意的是：`To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 10−3.（为了加速S=384网络的训练，我们使用 $S=256$ 预训练好的权重来初始化它，并且使用了更小的初始学习率10-3。）`这也是**微调（Fine-tuning）**的一种体现，思想值得借鉴。

#### 多尺度训练（Multi-Scale Training）

>The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range [Smin, Smax] (we used Smin = 256 and Smax = 512). Since objects in images can be of different size, it is beneficial to take this into account during training. This can also be seen as training set augmentation by scale jittering, where a single model is trained to recognise objects over a wide range of scales. For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 384.

同样使用了微调技巧。

对上面的策略进行分析总结，单/多尺度训练跟随机裁剪策略一样，为训练数据增加了目标物体位置多样性，强迫模型学习目标物体局部特征。此策略还增加了一个尺度多样性，这更符合模型部署测试时的情形，因为用户传入的图像中的目标物体大小是不确定的。如果模型训练的时候目标物体是一个固定尺寸，鲁棒性很差，当输入图像尺寸较训练尺寸有较大差异时模型就会出现精度下降或者完全不可用的窘境。当单/多尺度训练结合随机裁剪训练策略，模型训练情形更加贴近实际部署测试，模型鲁棒性大大提高。而上述的单尺度训练（Single-Scale Training）和多尺度训练（Multi-Scale Training）进行效果对比，显然后者效果更加（尺度灵活性更好），这一点在论文后面的章节----CLASSIFICATION EXPERIMENTS有数据体现。

---

## 密集评估（Dense Evaluation）VS  多裁剪块评估（Multi-Crop Evaluation）

### 密集评估（Dense Evaluation）

测试阶段将原始图像直接送入网络进行推理，不作224x224大小的强制要求。但是由于原网络的全连接层的存在，其强制限制输入图像的大小必须为224x224，所以原网络做了一个重要的变换，将第一个全连接层换成了7x7卷积层，后面的两个全连接层都换成1x1卷积层，通道数变化为**4096（低级特征）-4096（高级特征）-1000（num_classes）**。正是这个变换，使得网络适配更大的输入图像尺寸范围。接着分析下变换后网络相较原始FC网络的优势：

- ***更小的参数量：***将全连接层转换成卷积层后，参数量减少，网络推理速度性能更优。同时原来全连接层的庞大参数很容易导致网络过拟合，参数量的减小有效缓解了这个问题。
- ***更加丰富的语义信息：***对于原FC网络一定要求输入图像为224x224大小，这也意味着为了满足输入要求，强制缩放或者裁剪图像的操作不可避免。这也会导致图像中的细节，有效信息模糊或者丢失，引入噪声。这对结果有明显消极影响。但是替换为卷积层后，输入图像尺寸允许动态，不会丢失细节和有效信息。同时由于卷积的感受野特性，让其拥有丰富的语义信息，对结果有明显正向影响。

需要特别注意的是：在输入为224x224大小的图像时，原始FC网络和转换后的网络是完全等效的，参数量也相等，输出也相同。

### 多裁剪块评估（Multi-Crop Evaluation）

Multi-Crop 操作是指按照固定规则（如四角、中心及其翻转）从测试图像中裁剪出多张局部小图（224x224），分别送入网络进行独立预测，最后对所有预测结果取平均值以得到最终分类结果。这一操作解决了关键目标丢失（随机裁剪可能丢失目标）的问题，能够小幅度的提升测试精度，但是计算开销较大。

### 对比（VS）

下文用DE，MC代称上述两种方法：

- ***计算效率：***由于MC会裁剪出多张局部小图，论文中提到：`for reference we also evaluate our networks using 50 crops per scale (5 × 5 regular grid with 2 flips), for a total of 150 crops over 3 scales, which is comparable to 144 crops over 4 scales used by Szegedy et al. (2014).`为了测试一张图就要输入150张局部小图进入网络计算，计算效率是十分低的。相反DE只需要输入一次，且不需要进行裁剪和缩放操作，效率相比十分高效。当然在GPU资源充足的理想情况下，有能力将150张局部小图打包成一个Batch送入GPU并行计算，MC效率会大大提高，可以大幅缩小与DE之间的效率差异，且同时带来精度提升。
- ***采样精度差异：***MC：采样细致，DE：采样相较粗糙。MC会裁剪大量的局部小图作为网络输入，这些局部小图之间存在大量重叠，既防止了关键目标或者关键特征遗漏（采样数量多，覆盖区域广），又将关键目标和关键特征多次送入网络推理（小图高度重叠），使得推理结果不受噪声和单次偶然误差的影响，所有局部小图结果平均，精度提高。DE由于是全卷积网络，卷积核固定的步长可能会导致关键特征遗漏丢失，采样较粗糙。且不具备MC那么强的抗噪，对关键特征多次推理“加强容错”的能力。
- ***卷积边界条件：***两者的边界条件是**互补**的：`multi-crop evaluation is complementary to dense evaluation due to different convolution boundary condition`。MC由于是在原图上裁剪局部子图送入网络，那么一定会伴随着语义的丢失，卷积特征图会填充0。DE由于是原图输入，那么相同裁剪区域的填充则是区域周围的原图部分。举个例子，当两个方法都在同一区域做卷积时，MC因为是裁剪的原因，它会丢失周围的图像信息，那么在卷积填充的时候只能补零，而且卷积核不具有除这个区域外的语义信息。DE是输入原图，所以它还保留了除这一区域外的语义信息，卷积核在原图上滑动的时候，就会保留周围区域的语义信息。

MC方法的精确度略高于DE方法。根据两种方法的卷积边界条件是互补的结论，**分别做MC和DE后取平均**是追求精确度最好方式，但是计算效率十分低。所以在实际的部署中采取DE是最好的方法。因为`While we believe that in practice the increased computation time of multiple crops does not justify the potential gains in accuracy`，MC多出来的计算时间对比起其提升的微小准确率是不划算的。

## 分类实验（Classification Experiments）

重要三点：

- 越深的网络，分类误差越小。更深的网络对于更大的数据集可能更有利。`we observe that the classification error decreases with the increased ConvNet depth`。`he error rate of our architecture saturates when the depth reaches 19 layers, but even deeper models might be beneficial for larger datasets.`
- 网络C比网络B更好，说明引入更多额外非线性是有利的。网络D比网络C好，说明增加感受野也很重要。`This indicates that while the additional non-linearity does help (C is better than B), it is also important to capture spatial context by using conv. filters with non-trivial receptive fields (D is better than C). `
- 深层网络配小卷积核优于浅层网络配大卷积核。`which confirms that a deep net with small filters outperforms a shallow net with larger filters.`
- LRN（局部响应归一化）并没有改善网络性能。`we note that using local response normalisation (A-LRN network) does not improve on the model A without any normalisation layers. We thus do not employ normalisation in the deeper architectures (B–E).`

---

## 代码实现（Code Implementation）

```python
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
```

<div align='right'> --Ljh  2026.2.5 </div>





