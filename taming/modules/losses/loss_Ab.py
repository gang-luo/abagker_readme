import torch
import torch.nn as nn

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

class CELoss(nn.Module):
    def __init__(self,
                 kd_weight,
                 koff_weight
                 ):
        super().__init__()
        ### loss weights 
        self.kd_weight=kd_weight
        self.koff_weight=koff_weight
        
        # 损失函数定义
        self.MSEloss = torch.nn.MSELoss(reduction='mean')
        self.MAEloss = torch.nn.L1Loss(reduction='mean')
        self.HBloss = torch.nn.HuberLoss(reduction='mean')
        
    
    def forward(self,train_type, kd_pre, koff_pre, kd, koff, aux_dict, optimizer_idx, split="train"): 
        
        if optimizer_idx==0:
            
            if "moe" in train_type:
                coeff_balance = 0.05 # 原先为0.01
                aux_loss = coeff_balance * aux_dict['importance_loss'] 

                loss_kd = self.MSEloss(kd_pre, kd)
                loss = loss_kd * self.kd_weight + aux_loss

                log = {"{}/loss".format(split): loss.clone().detach(),
                        "{}/kd_loss".format(split): loss_kd.clone().detach(),
                        "{}/aux_loss".format(split): aux_loss.clone().detach()
                        }

                
        return loss,log
    
