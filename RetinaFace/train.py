import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import cv2
from data import preproc,WiderFaceDataset, detection_collate
from model import RetinaFace
from layers import MultiBoxLoss
from layers import PriorBox
from model import cfg_mnet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True 

    img_dim = cfg_mnet['image_size'][0]
    rgb_means = (104.0, 117.0, 123.0)
    batch_size = cfg_mnet['batch_size']
    
    dataset = WiderFaceDataset(
        txt_path='./data/widerface/train/label.txt', 
        preproc=preproc(img_dim=img_dim, rgb_means=rgb_means)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=detection_collate, 
        pin_memory=False 
    )

    net = RetinaFace(cfg=cfg_mnet).to(device)
    
    criterion = MultiBoxLoss(
        num_classes = 2,
        overlap_threshold = 0.35,
        neg_pos_ratio = 7,
        variances = cfg_mnet['variance'] 
    )
    
    priorbox = PriorBox(cfg=cfg_mnet)
    with torch.no_grad():
        priors = priorbox.forward().to(device)

    initial_lr = 1e-3
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)

    net.train()
    max_epoch = 100
    epoch_size = len(dataset) // batch_size
    step_index = 0

    for epoch in range(max_epoch):
 
        if epoch in [55, 68]:
            step_index += 1
            current_lr = initial_lr * (0.1 ** step_index)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"学习率物理降温 -> {current_lr}")

        start_time = time.time()
        
        for iteration, (images, targets) in enumerate(dataloader):

            images = images.to(device, non_blocking=True)
            targets = [anno.to(device, non_blocking=True) for anno in targets]

            out = net(images)

            optimizer.zero_grad()
            loss_l, loss_c, loss_lm = criterion(out, priors, targets)
            
            loss = 2.0 * loss_l + 1.0 * loss_c + 1.0 * loss_lm

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"Epoch:{epoch:03d} || Iter:{iteration:04d}/{epoch_size} || "
                      f"Loc:{loss_l.item():.4f} Conf:{loss_c.item():.4f} Landm:{loss_lm.item():.4f} || "
                      f"Total:{loss.item():.4f}")

        print(f"--- Epoch {epoch} 镇压完成 --- 耗时: {time.time() - start_time:.2f}s")
        
        if epoch % 5 == 0 or epoch == max_epoch - 1:
            save_path = f'./weights/mobilenet0.25_epoch_{epoch}.pth'
            torch.save(net.state_dict(), save_path)
            print(f"物理快照已写入硬盘: {save_path}")

if __name__ == '__main__':
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    train()