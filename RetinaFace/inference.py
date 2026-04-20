import torch
import model
import cv2
import numpy as np
import layers
import time
import utils
import torchvision

def inference():
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = model.RetinaFace(cfg=model.cfg_mnet,phase='test')
    net = net.to(device)

    weight_path = './weights/mobilenet0.25_epoch_99.pth'
    state_dict = torch.load(weight_path,map_location=device,weights_only=True)
    net.load_state_dict(state_dict)

    net.eval()

    image_path = './image/test.jpg'
    image_raw = cv2.imread(image_path)
    if image_raw is None:
        print('image path error')

    #image_raw = cv2.resize(image_raw,(640,640))
    image  = np.float32(image_raw)
    height,width,_ = image.shape
    image -= (104,117,123)
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    cfg_mnet = model.cfg_mnet
    cfg_mnet['image_size'] = [height,width]

    priorbox = layers.PriorBox(cfg=cfg_mnet)
    priors = priorbox.forward().to(device)

    start_time = time.time()

    conf,loc,landms = net(image)
    print('forward pass consume:',time.time() - start_time,'s')

    confidence_threshold = 0.5

    scores = conf.squeeze(0)[:,1]
    mask = scores > confidence_threshold

    loc = loc.squeeze(0)[mask]
    landms = landms.squeeze(0)[mask]
    priors = priors[mask]
    scores = scores[mask]

    if loc.shape[0] == 0:
        print('no face')
        return
    
    boxes = utils.decode(loc=loc,priors=priors,variances=cfg_mnet['variance'])
    lms = utils.landms_decode(pred=landms,priors=priors,variances=cfg_mnet['variance'])

    scale_boxes = torch.Tensor([width,height,width,height]).to(device)
    boxes = boxes * scale_boxes

    scale_landms = torch.Tensor([
        width,height,width,height,width,height,width,height,width,height
    ]).to(device)
    lms = lms * scale_landms

    keep_index = torchvision.ops.nms(boxes,scores,0.4)

    boxes = boxes[keep_index].cpu().numpy()
    lms = lms[keep_index].cpu().numpy()
    scores = scores[keep_index].cpu().numpy()

    for i,box in enumerate(boxes):
        x_min,y_min,x_max,y_max = map(int,box)
        score = scores[i]

        cv2.rectangle(image_raw,(x_min, y_min),(x_max, y_max),(0, 255, 0),2)
        cv2.putText(image_raw, f"{score:.3f}", (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pts = lms[i].reshape(-1, 2)
        for pt in pts:
            cv2.circle(image_raw,(int(pt[0]),int(pt[1])),2,(0, 0, 255),2)
    
    cv2.imwrite('result.jpg',image_raw)

if __name__ == '__main__':
    inference()
    





