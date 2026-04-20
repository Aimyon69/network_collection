import torch
import torch.nn as nn
from typing import List
from utils import match
import torch.nn.functional as F

class MultiBoxLoss(nn.Module):
    def __init__(
            self,
            num_classes: int,
            overlap_threshold: float,
            neg_pos_ratio: int,
            variances: List[float]
    ) -> None:
        super(MultiBoxLoss,self).__init__()
        
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variances = variances

    def forward(self,predictions: torch.Tensor,priors: torch.Tensor,targets: List[torch.Tensor]) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        conf_data,loc_data,landm_data = predictions

        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]

        device = loc_data.device
        priors = priors.to(device=device) # priors is just a label , can not influence the outer variable
        targets = [t.to(device) for t in targets] # the same as the above

        loc_t = torch.zeros(batch_size,num_priors,4,dtype=torch.float32,device=device)
        conf_t = torch.zeros(batch_size,num_priors,dtype=torch.long,device=device)
        landm_t = torch.zeros(batch_size,num_priors,10,dtype=torch.float32,device=device)

        for idx in range(batch_size):
            truths = targets[idx][:,:4]
            landms = targets[idx][:,4:14]
            labels = targets[idx][:,-1].long()
            match(
                threshold = self.overlap_threshold,
                truths = truths,
                priors = priors,
                variances = self.variances,
                labels = labels,
                landms = landms,
                loc_t = loc_t,
                conf_t = conf_t,
                landm_t = landm_t,
                idx = idx 
            )
        
        pos = conf_t > 0 
        nums_pos_total = torch.clamp(pos.sum().float(),min=1.0)
        nums_pos_per_image = pos.sum(dim=1,keepdim=True) 
    
        pos_box_preds = loc_data[pos] 
        pos_box_gt = loc_t[pos] 
        loss_box = F.smooth_l1_loss(pos_box_preds,pos_box_gt,reduction='sum')
      
        pos_landm_preds = landm_data[pos]
        pos_landm_gt = landm_t[pos]
        loss_landm = F.smooth_l1_loss(pos_landm_preds,pos_landm_gt,reduction='sum')
     
        batch_conf = conf_data.view(-1,self.num_classes)
        loss_c = F.cross_entropy(batch_conf,conf_t.view(-1),reduction='none',ignore_index=-1)
        loss_c[pos.view(-1)] = 0.0
        ignored = (conf_t < 0).view(-1) 
        loss_c[ignored] = 0.0
        loss_c = loss_c.view(batch_size,-1)
        _,loss_idx = loss_c.sort(dim=1,descending=True)
        _,idx_rank = loss_idx.sort(dim=1)
        num_neg = torch.clamp(self.neg_pos_ratio * nums_pos_per_image,max=pos.shape[1] - 1)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)
        pos_neg_mask = pos | neg_mask
        c_preds = conf_data[pos_neg_mask]
        c_gt = conf_t[pos_neg_mask]
        loss_c_final = F.cross_entropy(c_preds,c_gt,reduction='sum')

        return loss_box / nums_pos_total,loss_c_final / nums_pos_total ,loss_landm / nums_pos_total



        

            
