import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from model import MobileFaceNet
from collections import OrderedDict

class LFWDataset(Dataset):
    def __init__(self, lfw_dir: str, pairs_path: str):
        self.lfw_dir = lfw_dir
        self.pairs = []
        self.labels = []
        
        with open(pairs_path, 'r') as f:
            lines = f.readlines()[1:] 
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:

                    name, id1, id2 = parts
                    path1 = os.path.join(lfw_dir, name, f"{name}_{int(id1):04d}.jpg")
                    path2 = os.path.join(lfw_dir, name, f"{name}_{int(id2):04d}.jpg")
                    self.pairs.append((path1, path2))
                    self.labels.append(1) 
                elif len(parts) == 4:

                    name1, id1, name2, id2 = parts
                    path1 = os.path.join(lfw_dir, name1, f"{name1}_{int(id1):04d}.jpg")
                    path2 = os.path.join(lfw_dir, name2, f"{name2}_{int(id2):04d}.jpg")
                    self.pairs.append((path1, path2))
                    self.labels.append(0) 
                else:
                    continue

    def __len__(self):
        return len(self.pairs)

    def _preprocess(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        img1 = self._preprocess(path1)
        img2 = self._preprocess(path2)
        return img1, img2, self.labels[idx]

def evaluate_lfw():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFaceNet().to(device)
    
    weights_path = './weights/mobilefacenet.pth'
    state_dict = torch.load(weights_path,weights_only=True)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[:7]
        if name == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.eval()
    
    dataset = LFWDataset(lfw_dir='./data/lfw/lfw-112x96', pairs_path='./data/lfw/pairs.txt')
  
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=8)
    
    similarities = []
    labels = []
    
    print("[*] Extracting features for 6000 pairs...")
    
    with torch.no_grad(): 
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            
            feat1 = model(img1)
            feat2 = model(img2)
            
            feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
            feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)

            cos_sim = (feat1 * feat2).sum(dim=1)
            
            similarities.extend(cos_sim.cpu().numpy())
            labels.extend(label.numpy())
            
    similarities = np.array(similarities)
    labels = np.array(labels)

    print("[*] Performing 10-Fold Cross Validation...")
    k_fold = KFold(n_splits=10, shuffle=False)
    
    accuracies = []
    thresholds = np.arange(-1.0, 1.0, 0.005) 
    best_threshold_list = []
    
    for train_idx, test_idx in k_fold.split(similarities):
        train_sims, train_labels = similarities[train_idx], labels[train_idx]
        test_sims, test_labels = similarities[test_idx], labels[test_idx]

        best_acc = 0.0
        best_thresh = 0.0
        for thresh in thresholds:
            preds = (train_sims > thresh).astype(int)
            acc = np.mean(preds == train_labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        best_threshold_list.append(best_thresh)

        test_preds = (test_sims > best_thresh).astype(int)
        test_acc = np.mean(test_preds == test_labels)
        accuracies.append(test_acc)
        
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    best_mean_thresh = np.mean(best_threshold_list) 
    
    print("========================================")
    print(f"LFW Verification Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f'best threshold : {best_mean_thresh}')
    print("========================================")

if __name__ == '__main__':
    evaluate_lfw()