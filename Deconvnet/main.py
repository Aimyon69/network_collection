import models
import utils
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import cv2  as cv
from torchvision.transforms import transforms
from functools import partial
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def load_image(img_path: str) -> torch.Tensor:
    try:
        img = cv.imread(img_path)
        img = cv.resize(img,(224,224))
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
        return None
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img.unsqueeze_(0)

    return img

def store(model: models.vgg_conv) -> None:
    def hook(
            module: nn.Module,
            input: torch.Tensor,
            output: torch.Tensor,
            key: int
    ) -> None:
        if isinstance(module,nn.MaxPool2d):
            model.feature_maps[key] = output[0]
            model.pool_locs[key] =  output[1]
        else:
            model.feature_maps[key] = output 
            
    for idx,layer in enumerate(model._modules.get('features')):
        layer.register_forward_hook(partial(hook,key=idx))

def layer_visualization(
        layer: int,
        vgg_conv: models.vgg_conv,
        vgg_deconv: models.vgg_deconv
) -> np.ndarray:
    if layer not in vgg_conv.feature_maps:
        raise ValueError('layer is not a conv')
    
    num_feat = vgg_conv.feature_maps[layer].shape[1]

    act_list: List[int] = []
    for i in range(0,num_feat):
        choose_map = vgg_conv.feature_maps[layer][0,i,:,:]
        activation = torch.max(choose_map)
        act_list.append(activation)
    act_list = np.array(act_list)
    mark = np.argmax(act_list)

    choose_map = vgg_conv.feature_maps[layer][0,mark,:,:]
    max_activation = torch.max(choose_map)
    new_map = torch.zeros_like(vgg_conv.feature_maps[layer])
    choose_map = torch.where(
        choose_map==max_activation,
        max_activation,
        0.0
    )
    new_map[0,mark,:,:] = choose_map
    with torch.no_grad():
        deconv_output = vgg_deconv(new_map,layer,vgg_conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1,2,0)
    new_img = (new_img - new_img.min()) / (new_img.max()-new_img.min()) * 255
    new_img = new_img.astype(np.uint8)

    return new_img

def func1(
        img_path: str,
        model_name: str,
        layers: List[int],
        save_path: str
) -> None:
    img = load_image(img_path)
    if img is None:
        print('img path error')
        return
    
    vgg_conv = models.vgg_conv(model_name)
    vgg_deconv = models.anynet_deconv(vgg_conv)

    for layer in layers:
        if layer not in vgg_deconv.conv2deconv_indices:
            print('layers indices error')
            return

    vgg_conv.eval()
    store(vgg_conv)
    with torch.no_grad():
        conv_output = vgg_conv(img)
    print('predicted: ',utils.decode_predictions(conv_output,top=1))

    n = len(layers) + 1
    rows = (n + 3) // 4
    cols = 4
    fig,axes = plt.subplots(rows,cols,figsize=(cols*3,rows*3))
    axes = axes.flatten()
    orginal_img = cv.imread(img_path)
    orginal_img = cv.resize(orginal_img,(224,224))
    axes[0].imshow(orginal_img[:,:,::-1])
    axes[0].set_title('original img')

    vgg_deconv.eval()
    for idx,layer in enumerate(layers):
        deconv_img = layer_visualization(layer,vgg_conv,vgg_deconv)
        axes[idx+1].imshow(deconv_img[:,:,::-1])
        axes[idx+1].set_title(f'{layer} layer')

    for i in range(n+1,rows*cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == '__main__':
    func1('data/cat.jpg','vgg16',[14,17,19,21,24,26,28],'result.jpg')


