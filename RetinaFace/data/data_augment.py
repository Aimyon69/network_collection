import numpy as np
import torch
import random
from utils import matrix_iof
import cv2

def _crop(
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        landms: np.ndarray,
        img_dim: int
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,bool]:
    height,width,_ = image.shape
    pre_scale = [0.3,0.45,0.6,0.8,1]

    for _ in range(250):
        scale = random.choice(pre_scale)
        h = w = int(scale * min(height,width))
        if w == width:
            lx = 0
        else:
            lx = random.randrange(width - w)
        if h == height:
            ly = 0
        else:
            ly = random.randrange(height - h)
        roi = np.array([lx,ly,lx + w,ly + h])

        overlap = matrix_iof(box_a=boxes,box_b=roi[None,:])
        flag = (overlap >= 1.0)
        if not flag.any():
            continue

        center = (boxes[:,2:] + boxes[:,:2]) / 2
        mask = (center[:,0] >= roi[None,0]) & (center[:,0] <= roi[None,2]) & (center[:,1] >= roi[None,1]) & (center[:,1] <= roi[None,3])
        boxes_filter = boxes[mask]
        labels_filter = labels[mask]
        landms_filter = landms[mask]
        
        boxes_filter[:,[0,2]] = np.clip(boxes_filter[:,[0,2]] - roi[None,0],0,w)
        boxes_filter[:,[1,3]] = np.clip(boxes_filter[:,[1,3]] - roi[None,1],0,h)

        landms_filter = landms_filter.reshape(-1,5,2)
        landms_filter -= roi[None,None,:2]
        landms_filter = landms_filter.reshape(-1,10)

        w_t = (boxes_filter[:,2] - boxes_filter[:,0]) / w * img_dim
        h_t = (boxes_filter[:,3] - boxes_filter[:,1]) / h * img_dim
        mask_valid = np.minimum(w_t,h_t) >= 16.0
        boxes_filter = boxes_filter[mask_valid]
        labels_filter = labels_filter[mask_valid]
        landms_filter = landms_filter[mask_valid]
        if boxes_filter.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3],roi[0]:roi[2]]

        return image_t,boxes_filter,labels_filter,landms_filter,False
    
    return image,boxes,labels,landms,True 

def _distort(image: np.ndarray) -> np.ndarray:
    def _convert(image: np.ndarray,alpha: float = 1,beta: float = 0) -> None:
        temp = image.astype(np.float32) * alpha + beta
        temp[temp < 0] = 0
        temp[temp > 255] = 255
        image[:] = temp

    image = image.copy()

    if random.randrange(2):
        if random.randrange(2):
            _convert(image=image,beta=random.uniform(-32,32))
        
        if random.randrange(2):
            _convert(image=image,alpha=random.uniform(0.5,1.5))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        if random.randrange(2):
                _convert(image=image[:,:,1],alpha=random.uniform(0.5,1.5))

        if random.randrange(2):
                temp = image[:,:,0].astype(int) + random.randint(-18,18)
                temp %= 180
                image[:,:,0] = temp

        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    else:
        if random.randrange(2):
            _convert(image=image,beta=random.uniform(-32,32))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            _convert(image=image[:,:,1],alpha=random.uniform(0.5,1.5))

        if random.randrange(2):
            temp = image[:,:,0].astype(int) + random.randint(-18,18)
            temp %= 180
            image[:,:,0] = temp

        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

        if random.randrange(2):
            _convert(image=image,alpha=random.uniform(0.5,1.5))

    return image
                     
def _mirror(image: np.ndarray,boxes: np.ndarray,landms: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    _,width,_ = image.shape

    if random.randrange(2):
        image = image[:, ::-1, :].copy()
        boxes = boxes.copy()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        landms = landms.copy()
        landms = landms.reshape([-1,5,2])
        landms[:,:,0] = width - landms[:,:,0]
        landms = landms[:, [1, 0, 2, 4, 3], :]
        landms = landms.reshape(-1,10)

        return image,boxes,landms
    return image,boxes,landms
    
def _pad_to_square(image: np.ndarray,rgb_means: float,pad_image_flag: bool) -> np.ndarray:
    if not pad_image_flag:
        return image
    
    height,width,_ = image.shape
    long_side = max(height,width)
    image_t = np.empty((long_side,long_side,3),dtype=image.dtype) 
    image_t[:,:] = rgb_means
    image_t[0:0 + height,0:0 + width] = image
    return  image_t

def _resize_subtract_mean(image: np.ndarray,img_dim: int,rgb_means: float) -> np.ndarray:
    interp_methods = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA,cv2.INTER_NEAREST,cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (img_dim, img_dim), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_means

    return image.transpose(2,0,1)

class preproc:
    def __init__(self,img_dim: int,rgb_means: float):
        self.img_dim = img_dim
        self.rgb_means = rgb_means


    def __call__(self,image: np.ndarray,targets: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        assert targets.shape[0] > 0

        boxes = targets[:,:4].copy()
        labels = targets[:,-1].copy()
        landms = targets[:,4:-1].copy()

        image_t,boxes_t,labels_t,landms_t,pad_image_flag = _crop(image=image,boxes=boxes,labels=labels,landms=landms,img_dim=self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image=image_t,rgb_means=self.rgb_means,pad_image_flag=pad_image_flag)
        image_t,boxes_t,landms_t = _mirror(image=image_t,boxes=boxes_t,landms=landms_t)
        height,width,_ = image_t.shape
        image_t = _resize_subtract_mean(image=image_t,img_dim=self.img_dim,rgb_means=self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landms_t[:, 0::2] /= width
        landms_t[:, 1::2] /= height
        targets_t = np.hstack((boxes_t,landms_t,labels_t[:,None]))

        return image_t,targets_t



    







    
