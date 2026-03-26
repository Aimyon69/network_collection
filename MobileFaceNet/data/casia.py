import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
from typing import List
import random

class CASIAFace(data.Dataset):
    def __init__(self,root: str):
        self.image_list = []
        self.label_list = []
        image_txt_path = os.path.join(root,'CASIA-WebFace-112X96.txt')
        image_dir = os.path.join(root,'CASIA-WebFace-112X96')

        with open(image_txt_path) as f:
            lines = f.read().splitlines()
            for line in lines:
                image_path,label = line.split(' ')
                self.image_list.append(os.path.join(image_dir,image_path))
                self.label_list.append(int(label))
                
        self.class_nums = len(np.unique(self.label_list))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = image.astype(np.float32)

        flip_flag = random.randrange(2)
        if flip_flag:
            image = image[:,::-1,:].copy()
        image = (image - 127.5) / 128
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image)

        return image,torch.tensor(self.label_list[index],dtype=torch.long)
