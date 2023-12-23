# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs
from torchmetrics.classification import BinaryHammingDistance
import torch
from torch import nn
import torch.nn.functional as F

def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
    
def conv3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False),
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
    )    


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper， better than L2(Mse loss)
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.eps = 0.1 # in case of gradient explode

    def forward(self, input, target, mask=None, interpolate=True):
        
        
        loss_sum = list()
        
        for input_, target_ in zip(input, target):
            target_.requires_grad_(False)
            g = torch.log(input_ + self.eps) - torch.log(target_ + self.eps)                          
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
            loss_sum.append(2 * torch.sqrt(Dg))
            
        return sum(loss_sum)
        
    def gradient_compute(self, log_pridiction, mask, log_gt):
        pass


class Connectors(nn.Module):
    #student feature as input
    def __init__(self):
        super().__init__()
        
        #self.s_t_pair = [(256, 512)]
        self.connector1_1 = conv1x1_bn(64, 64)   #for s, t in self.s_t_pair
        self.connector2_1 = conv3x3(64, 64)   #for s, t in self.s_t_pair

        self.connector1_2 = conv1x1_bn(64, 64)   #for s, t in self.s_t_pair
        self.connector2_2 = conv3x3(64, 64)   #for s, t in self.s_t_pair
        
        self.connector1_3 = conv1x1_bn(64, 64)   #for s, t in self.s_t_pair
        self.connector2_3 = conv3x3(64, 64)   #for s, t in self.s_t_pair        

        self.connector1_4 = conv1x1_bn(128, 128)   #for s, t in self.s_t_pair
        self.connector2_4 = conv3x3(128, 128)   #for s, t in self.s_t_pair
        
        self.connector1_5 = conv1x1_bn(256, 256)   #for s, t in self.s_t_pair
        self.connector2_5 = conv3x3(256, 256)   #for s, t in self.s_t_pair        

        self.connector1_6 = conv1x1_bn(256, 256)   #for s, t in self.s_t_pair
        self.connector2_6 = conv3x3(512, 512)   #for s, t in self.s_t_pair    
                                
    def forward_no_norm(self, x):
        
        x1, x2, x3, x4, x5, x6 = x
        x1 = self.connector2_1(x1)
        x2 = self.connector2_2(x2)        
        x3 = self.connector2_3(x3)        
        x4 = self.connector2_4(x4)
        x5 = self.connector2_5(x5)      
        x6 = self.connector2_6(x6)          
        
        return x
    
    def forward_norm(self, x):

        x1, x2, x3, x4, x5, x6 = x
        x1 = self.connector1_1(x1)
        x2 = self.connector1_2(x2)        
        x3 = self.connector1_3(x3)        
        x4 = self.connector1_4(x4)
        x5 = self.connector1_5(x5)      
        x6 = self.connector1_6(x6)          
               
        return x
    
    def forward(self, x, flag):
        if flag == "f":
            return self.forward_no_norm(x)
        elif flag == 'a':
            return self.forward_norm(x)


class Spik_SelfSupModel(SfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        # Initializes SfmModel
        super().__init__(**kwargs)
        # Initializes the photometric loss
        self._photometric_loss = MultiViewPhotometricLoss(**kwargs)
        
        self.spike_connector_feature = Connectors()
        #self.connector_activation = Connectors(flag = "a")
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, poses,
                             intrinsics, return_logs=False, progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._photometric_loss(
            image, ref_images, inv_depths, intrinsics, intrinsics, poses,
            return_logs=return_logs, progress=progress)

    def feat_l2_loss(self, spike_features, rgb_features):   

        all_losses = []
        for spike_feat, rgb_feat in zip(spike_features, rgb_features):
            all_losses.append(F.mse_loss(spike_feat, rgb_feat, size_average=True))
        
        return 0.1 * sum(all_losses)
        

    def act_ham_loss(self, spike_features, rgb_features):
        
        act_loss_func = BinaryHammingDistance(threshold=0.5, multidim_average='global', ignore_index=None, validate_args=True)
        all_losses = []
        for spike_feat, rgb_feat in zip(spike_features, rgb_features):
            all_losses.append(act_loss_func(spike_feat, rgb_feat, size_average=True))
        
        return 0.1 * sum(all_losses)    
    
        
    def rgb2spike_distillation_loss(self, spike_features, rgb_features):
        
        spike_features_out = self.spike_connector_feature(spike_features, 'f')
        loss1 = self.feat_l2_loss(spike_features_out, rgb_features)       
        
        rgb_features_act = self.relu(rgb_features - torch.mean(rgb_features))
        spike_activation_out = self.spike_connector_feature(spike_features, 'a')
        act_loss = BinaryHammingDistance(threshold=0.5, multidim_average='global', ignore_index=None, validate_args=True)
        loss2 = self.act_ham_loss(spike_activation_out, rgb_features)  
        
        loss = loss1 + loss2
        
        return loss
        
    def spike_self_distillation_label_loss(self, spike_pred_list, fusion_pred_list):
        
        label_loss_func = SILogLoss()
        label_loss = label_loss_func(spike_pred_list, fusion_pred_list)
               
        return label_loss
        
           
    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], output['poses'], batch['intrinsics'],
                return_logs=return_logs, progress=progress)

            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }


    def forward_distill(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        rgb_branch = output['encoding']['fusion_features']
        spike_branch = output['encoding']['spike_features']
        fusion_pred = output['inv_depths']
        
        output_single = super().forward_single(batch, return_logs=return_logs)
        spike_pred = output_single['inv_depths']
        
        
        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            pass
               
            self_distill_feature_loss = self.rgb2spike_distillation_loss(rgb_branch, spike_branch)
            self_distill_label_loss = self.spike_self_distillation_label_loss(spike_pred, fusion_pred)
            distill_loss = self_distill_feature_loss + self_distill_label_loss 
            # Return loss and metrics
            return {
                'distill_loss': distill_loss,
                **merge_outputs(output, self_sup_output),
            }
            
            