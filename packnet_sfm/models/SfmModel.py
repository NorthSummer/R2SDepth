# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.models.base_model import BaseModel
from packnet_sfm.models.model_utils import flip_batch_input, flip_output, upsample_output, flip_batch_spike_input
from packnet_sfm.utils.misc import filter_dict

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

        self.connector1_6 = conv1x1_bn(512, 512)   #for s, t in self.s_t_pair
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
        #x = self.connector1(x)
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

class SfmModel(BaseModel):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = 0.0 #flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps

        self.spike_connector_feature = Connectors()
        #self.connector_activation = Connectors(flag = "a")
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self._network_requirements = [
            'depth_net',
            'pose_net',
        ]

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net

    def depth_net_flipping(self, batch, flip):
        """
        Runs depth net with the option of flipping

        Parameters
        ----------
        batch : dict
            Input batch
        flip : bool
            True if the flip is happening

        Returns
        -------
        output : dict
            Dictionary with depth network output (e.g. 'inv_depths' and 'uncertainty')
        """
        # Which keys are being passed to the depth network
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        batch_input_spike = {key: batch[key] for key in filter_dict(batch, self._input_keys_spike)}
        #print(batch)
        #print(**batch_input_spike)
        if flip:
            # Run depth network with flipped inputs
            output, _ = self.depth_net(**flip_batch_input(batch_input), spike=flip_batch_spike_input(batch_input_spike)['spike_sequence']) #**flip_batch_input(batch_input_spike))
            # Flip output back if training
            output, _ = flip_output(output)
        else:
            # Run depth network
            output, _ = self.depth_net(**batch_input, spike=batch['spike_sequence'])#**batch_input_spike)
        return output

    def compute_depth_net(self, batch, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        output = self.depth_net_flipping(batch, flag_flip_lr)
        # If upsampling depth maps at training time
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output
        
    def compute_spike_depth_net(self, batch, epoch, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        #flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        #output = self.depth_net_flipping(batch, flip=False)
        # If upsampling depth maps at training time
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        batch_input_spike = {key: batch[key] for key in filter_dict(batch, self._input_keys_spike)}        
        output, encodings = self.depth_net(**batch_input, spike=batch['spike_sequence'], epoch=epoch)
        #rgb_output = encodings["teacher_features"]
        #spike_output = encodings["spike_features"]
        
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output, encodings        
        
    def compute_single_depth_net(self, batch):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        #flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        #output = self.depth_net_flipping(batch, flip=False)
        # If upsampling depth maps at training time
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        batch_input_spike = {key: batch[key] for key in filter_dict(batch, self._input_keys_spike)}     
        
        s_features = self.depth_net.pass_spike_features(spike=batch['spike_sequence'])
        c_features = self.spike_connector_feature(s_features, 'f')
        
        #output_single, spike_features = self.depth_net.forward_single_branch(**batch_input, spike=batch['spike_sequence'], connector=c_features)
        output_single = self.depth_net.forward_single_branch(spike=batch['spike_sequence'])
        
        if self.training and self.upsample_depth_maps:
            output_single = upsample_output(output_single, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output_single      

    def compute_pose_net(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, epoch, return_logs=False, force_flip=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        force_flip : bool
            If true, force batch flipping for inverse depth calculation

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """
        # Generate inverse depth predictions
        depth_output, encodings = self.compute_spike_depth_net(batch, epoch=epoch, force_flip=force_flip)
        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            **depth_output,
            'poses': pose_output,
            'encoding': encodings,
        }
    
    def forward_single(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        force_flip : bool
            If true, force batch flipping for inverse depth calculation

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """
        # Generate inverse depth predictions
        depth_output = self.compute_single_depth_net(batch)
        #spike_imitation = self.
        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            **depth_output,
            'poses': pose_output,
        }
    