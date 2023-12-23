from abc import abstractmethod

import torch.nn as nn

import os
import copy
from math import fabs, ceil, floor

import numpy as np
import torch
from torch.nn import ZeroPad2d

from packnet_sfm.networks.depth.submodules import ConvGRU, ConvLayer, ConvLayer_, ConvLeaky, ConvLeakyRecurrent, ConvRecurrent

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
    
    
    
        
class FireNet(BaseModel):
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

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        ff_act, rec_act = unet_kwargs["activations"]
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[0])

        self.G1 = self.rec_neuron(
            64, 64, kernel_size, activation=rec_act, **self.kwargs[1]
        )
        self.R1a = self.ff_neuron(
            64, 64, kernel_size, stride=2, activation=ff_act, **self.kwargs[2]
        )
        self.R1b = self.ff_neuron(
            64, 64, kernel_size, stride=2, activation=ff_act, **self.kwargs[3]
        )

        self.G2 = self.rec_neuron(
            64, 64, kernel_size, activation=rec_act, **self.kwargs[4]
        )
        self.R2a = self.ff_neuron(
            64, 128, kernel_size, stride=2, activation=ff_act, **self.kwargs[5]
        )
        self.R2b = self.ff_neuron(
            128, 256, kernel_size, stride=2, activation=ff_act, **self.kwargs[6]
        )
        self.R3b = self.ff_neuron(
            256, 512, kernel_size, stride=2, activation=ff_act, **self.kwargs[6]
        )
        #self.pred = ConvLayer(
        #    base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred
        #)

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
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
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
        x = 2.0 * torch.sum(x, dim=1, keepdim=True)


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
        x4, self._states[3] = self.R1b(x3, self._states[3])

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6])
        x7, self._states[7] = self.R2b(x6, self._states[7])
        
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

        return [x7] #{"spike_feature": [x7], "activity": activity}        



class FireNet_(BaseModel):
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

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs["num_bins"]
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        ff_act, rec_act = unet_kwargs["activations"]
        if type(unet_kwargs["spiking_neuron"]) is dict:
            for kwargs in self.kwargs:
                kwargs.update(unet_kwargs["spiking_neuron"])

        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, stride=1, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels,  kernel_size)
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, stride=2, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels,  kernel_size, stride=2, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels,  kernel_size)
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels*2,  kernel_size, stride=2, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)
        self.R2b = self.ff_neuron(
            base_num_channels*2, base_num_channels*4,  kernel_size, stride=2, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)

        self.R3b = self.ff_neuron(
            base_num_channels*4, base_num_channels*8,  kernel_size, stride=2, padding=1, activation='relu', norm=None,
                 BN_momentum=0.1)

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
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
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

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

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
        x4, self._states[3] = self.R1b(x3, self._states[3])

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6])
        x8, self._states[7] = self.R3b(x7, self._states[7])
        
        
        

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
                "8:R3b",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, x8]):
                activity[n] = l.detach().ne(0).float().mean().item()
        else:
            activity = None

        return [x1, x3, x4, x6, x7, x8] 



class LIFFireNet(FireNet):
    """
    Spiking FireNet architecture of LIF neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLIF
    ff_neuron = ConvLIF
    rec_neuron = ConvLIFRecurrent
    residual = False
    w_scale_pred = 0.01
  
  
class RNNFireNet(FireNet_):
    """
    Recurrent FireNet architecture of convolutional neurons for dense optical flow estimation from events.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvRecurrent
    residual = False        