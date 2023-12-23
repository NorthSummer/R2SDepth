from abc import abstractmethod

import torch.nn as nn

import os
import copy
from math import fabs, ceil, floor

import numpy as np
import torch
from torch.nn import ZeroPad2d

from packnet_sfm.networks.depth.submodules import ConvGRU, ConvLayer, ConvLayer_, ConvLeaky, ConvLeakyRecurrent, ConvRecurrent
from packnet_sfm.networks.layers.packnet.layers01 import Conv2D, SPackLayerConv3d


from packnet_sfm.networks.depth.submodules_spiking import (
    ConvALIF,
    ConvALIFRecurrent,
    ConvLIF,
    ConvLIFRecurrent,
    ConvPLIF,
    ConvPLIFRecurrent,
    ConvXLIF,
    ConvXLIFRecurrent,
)


class CropParameters:
    """
    Helper class to compute and store useful parameters for pre-processing and post-processing
    of images in and out of E2VID.
    Pre-processing: finding the best image size for the network, and padding the input image with zeros
    Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d(
            (
                self.padding_left,
                self.padding_right,
                self.padding_top,
                self.padding_bottom,
            )
        )

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0 : self.iy1, self.ix0 : self.ix1]


def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, "clone"):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print("{} is not iterable and has no clone() method.".format(tensor))


def copy_states(states):
    """
    Simple deepcopy if list of Nones, else clone.
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)
    
    
class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    
class STBaseModule(nn.Module):

    head_neuron = ConvALIF #ConvLIF
    ff_neuron = ConvALIF #ConvLIF #ConvLIF
    rec_neuron = ConvALIFRecurrent #ConvLIFRecurrent
    #mod_neuron = SPackLayerConv3d
    residual = False
    w_scale_pred = 0.01
    
    
    def __init__(self, input_channel, hidden_channel, out_channel, kernel_s, kernel_groups, stride_, padding_groups, activations, spiking_kwargs):
        super().__init__()
        ff_act, rec_act = activations
        self.head = self.head_neuron(input_channel, hidden_channel, kernel_s, stride_, ff_act, **spiking_kwargs)    
        
        self.mod_neuron = SPackLayerConv3d(in_channels=input_channel, out_channels=out_channel, kernel_size=3)
        self.num_branches = len(kernel_groups) + 1
        
        self.branch1 = self.ff_neuron(
            hidden_channel, out_channel // 4, kernel_groups[0], stride=1, activation=ff_act, **spiking_kwargs
        )
        self.branch2 = self.ff_neuron(
            hidden_channel, out_channel // 4, kernel_groups[1], stride=1, activation=ff_act, **spiking_kwargs
        )       
        self.branch3 = self.ff_neuron(
            hidden_channel, out_channel // 4, kernel_groups[2], stride=1, activation=ff_act, **spiking_kwargs
        )
        self.branch4 = self.ff_neuron(
            hidden_channel, out_channel // 4, kernel_groups[3], stride=1, activation=ff_act, **spiking_kwargs
        )
                
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.a_conv = nn.Conv1d(1, 1, 3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        
    def SE_Attention(self, input_m):
        
        y = self.avg_pool(input_m)
        y = self.a_conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return input_m * y.expand_as(input_m)        

        
    def forward(self, x, prev_state, residual=0):
        
        if prev_state is None:
            prev_state = [None] * self.num_branches #[torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)] * self.num_branches

        z = self.mod_neuron(x)
                                   
        x, prev_state[0] = self.head(x, prev_state[0])
                
        x1, prev_state[1] = self.branch1(x, prev_state[1], residual)
        x2, prev_state[2] = self.branch2(x, prev_state[2], residual)
        x3, prev_state[3] = self.branch3(x, prev_state[3], residual)
        x4, prev_state[4] = self.branch4(x, prev_state[4], residual)
                
        x_cat = torch.cat([x1,x2], dim=1)
        x_cat = torch.cat([x_cat,x3], dim=1)
        x_cat = torch.cat([x_cat,x4], dim=1)
        
        x = self.SE_Attention(x_cat)
        #print(x.shape, z.shape)
        x = torch.mul(x, z)
        
        
        return x, prev_state
        
        
        
class STNet(BaseModel):
    """
    FireNet architecture (adapted for optical flow estimation), as described in the paper "Fast Image
    Reconstruction with an Event Camera", Scheerlinck et al., WACV 2020.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvGRU
    residual = False
    num_recurrent_units = 8
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = None
    kernel_groups = [1, 3, 5, 7]
    padding_groups = [0, 1, 2, 3]
    
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        stride = unet_kwargs["stride"]
        self.base_num_channels = unet_kwargs["base_num_channels"]
        out_channel_ = unet_kwargs["out_channel"]
        kernel_size = unet_kwargs["kernel_size"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        ff_act, rec_act = unet_kwargs["activations"]

        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        self.head = self.head_neuron(self.num_bins, self.base_num_channels, kernel_size, activation=ff_act, **self.kwargs[0])
        #64, 64, 128, 256, 512 
        
        self.G1 = self.rec_neuron(
            self.base_num_channels, self.base_num_channels, kernel_size, activation=rec_act, **self.kwargs[1]
        )
        

        self.R1a = self.ff_neuron(
            input_channel = self.base_num_channels, hidden_channel = self.base_num_channels, out_channel = self.base_num_channels ,
            kernel_s = kernel_size, kernel_groups = self.kernel_groups, 
            stride_ = stride, padding_groups = self.padding_groups, activations=[ff_act, rec_act], spiking_kwargs = self.kwargs[2]
        )
        
        self.R1b = self.ff_neuron(
            input_channel = self.base_num_channels , hidden_channel = self.base_num_channels, out_channel = self.base_num_channels ,
            kernel_s = kernel_size, kernel_groups = self.kernel_groups, 
            stride_ = stride, padding_groups = self.padding_groups, activations=[ff_act, rec_act], spiking_kwargs = self.kwargs[3]
        )

        self.G2 = self.rec_neuron(
            self.base_num_channels , self.base_num_channels , kernel_size, activation=rec_act, **self.kwargs[4]
        )
        self.R2a = self.ff_neuron(
            input_channel = self.base_num_channels , hidden_channel = self.base_num_channels, out_channel = self.base_num_channels * 2, 
            kernel_s = kernel_size, kernel_groups = self.kernel_groups, 
            stride_ = stride, padding_groups = self.padding_groups, activations=[ff_act, rec_act], spiking_kwargs = self.kwargs[5]
        )
        self.R2b = self.ff_neuron(
            input_channel = self.base_num_channels * 2, hidden_channel = self.base_num_channels, out_channel = self.base_num_channels * 4, 
            kernel_s = kernel_size, kernel_groups = self.kernel_groups, 
            stride_ = stride, padding_groups = self.padding_groups, activations=[ff_act, rec_act], spiking_kwargs = self.kwargs[6]
        )
        self.R3b = self.ff_neuron(
            input_channel = self.base_num_channels * 4, hidden_channel = self.base_num_channels, out_channel = self.base_num_channels * 8, 
            kernel_s = kernel_size, kernel_groups = self.kernel_groups, 
            stride_ = stride, padding_groups = self.padding_groups, activations=[ff_act, rec_act], spiking_kwargs = self.kwargs[7]
        )

        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            detached_substates = []
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                #print(len(state))
                for sub_state in state:
                    detached_substates.append(sub_state.detach())
                detached_states.append(detached_substates) #(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward_events(self, event_voxel, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        x = event_voxel

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)
        x8, self._states[7] = self.R3b(x7, self._states[7])
        #print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape, x7.shape)
        #flow = self.pred(x7)

        # log activity
        if log:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:R1b",
                "5:G2",
                "6:R2a",
                "7:R2b",
                "8:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()
        else:
            activity = None
        #print(x7.shape, x6.shape, x4.shape, x3.shape)
        return [x1, x3, x4, x6, x7, x8]            #{"spike_feature": [x7], "activity": activity}        




class LIFSTNet(STNet):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLIF
    ff_neuron = STBaseModule #ConvLIF
    rec_neuron = ConvLIFRecurrent
    residual = False
    w_scale_pred = 0.01

class ALIFSTNet(STNet):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvALIF
    ff_neuron = STBaseModule #ConvLIF
    rec_neuron = ConvALIFRecurrent
    residual = False
    w_scale_pred = 0.01            