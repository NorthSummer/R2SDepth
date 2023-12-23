# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from packnet_sfm.networks.layers.packnet.layers01 import \
    PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
    
from packnet_sfm.networks.depth.StateNet_Encoder import StateNetPhasedRecurrent, SpikingStateNetPhasedRecurrent
from packnet_sfm.networks.depth.FireNet_Encoder import LIFFireNet, RNNFireNet
from packnet_sfm.networks.depth.STNet_Encoder import LIFSTNet, ALIFSTNet


def encoder_builder(type):
    
    if type == "StateNet":
        
        model_dict = dict(
            #num_bins_rgb = 1,
            #num_bins_events = 5,
            num_input_channels_rgb = 3, 
            num_input_channels_events = 1, 
            num_output_channels=1,
            skip_type = "sum",
            recurrent_block_type = "conv",
            state_combination = "convgru",
            activation='sigmoid', #spatial_resolution = [112,112],
            num_encoders = 6,
            base_num_channels = 64,
            num_residual_blocks = 2,
            use_upsample_conv = True,
            norm = "none"
        )        
           
        encoder = StateNetPhasedRecurrent(**model_dict)
        
    elif type == "Spiking-StateNet":

        model_dict = dict(
            #num_bins_rgb = 1,
            #num_bins_events = 5,
            num_input_channels_rgb = 3, 
            num_input_channels_events = 128, 
            num_output_channels=1,
            skip_type = "sum",
            recurrent_block_type = "conv",
            state_combination = "spiking-convgru",
            activation='sigmoid', #spatial_resolution = [112,112],
            num_encoders = 4,
            base_num_channels = 32,
            num_residual_blocks = 2,
            use_upsample_conv = True,
            norm = "none"
        )          
        encoder = SpikingStateNetPhasedRecurrent(**model_dict)

    elif type == "FireNet":
        
        model_dict = dict(
        encoding= "voxel", # voxel/cnt
        round_encoding= False, # for voxel encoding
        norm_input= False, # normalize input
        num_bins= 128,
        base_num_channels= 64,
        kernel_size= 3,
        activations= ["relu", "Null"], # activations for ff and rec neurons
        mask_output= True,
        spiking_neuron = "Null",
        )          
        encoder = RNNFireNet(model_dict)

    elif type == "Spiking-FireNet":
        
        spiking_neuron_ = dict(
            leak = [-4.0, 0.1],
            thresh = [0.8, 0.1],
            learn_leak = True,
            learn_thresh = True,
            hard_reset = True,
            )
        
        model_dict = dict(
            #name=LIFFireNet # for other models available, see models/model.py
            encoding= "voxel", # voxel/cnt
            stride = 2, #round_encoding= False, # for voxel encoding
            norm_input= False, # normalize input
            num_bins= 4,
            base_num_channels= 32,
            kernel_size= 3,
            activations= ["arctanspike", "arctanspike"], # activations for ff and rec neurons
            out_channel = 96, 
            mask_output= True,
            spiking_neuron = spiking_neuron_,
          )       
          
          
        encoder = IFFireNet(model_dict)

    
    elif type == "Spiking-STNet":
        
        spiking_neuron_lif = dict(
            leak = [-4.0, 0.1],
            thresh = [0.8, 0.1],
            learn_leak = True,
            learn_thresh = True,
            hard_reset = True,
            )

        spiking_neuron_alif = dict(
            leak_v=(-4.0, 0.1),
            leak_t=(-4.0, 0.1),
            t0=(0.01, 0.0),
            t1=(1.8, 0.0),
            learn_leak=True,
            learn_thresh=True,
            hard_reset=False,
            )
                    
        model_dict = dict(
            #name=LIFFireNet # for other models available, see models/model.py
            encoding= "voxel", # voxel/cnt
            stride = 2, #round_encoding= False, # for voxel encoding
            norm_input= False, # normalize input
            num_bins= 1,
            base_num_channels= 64,
            kernel_size= 3,
            activations= ["arctanspike", "arctanspike"], # activations for ff and rec neurons
            out_channel = 96, 
            mask_output= True,
            spiking_neuron = spiking_neuron_alif,
          )       
          
          
        encoder = ALIFSTNet(model_dict)
          
    return encoder


class TAM_module(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state, new_state


class Spike_PackNet01(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).

    https://arxiv.org/abs/1905.02693

    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, forward_single=False, **kwargs):
        super().__init__()
        self.version = version[1]
        self.version_II = version[3]
        self.version_III = version[5] if version[4]=='3' else None
        
        self.forward_single=forward_single
        # Input/output channels
        in_channels = 1
        out_channels = 1
        # Hyper-parameters
        ni, no = 64, out_channels
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512  #256
        sn1, sn2, sn3, sn4, sn5 = 64, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        # Initial convolutional layer
        
        self.spike_encoder = encoder_builder("Spiking-STNet") #encoder_builder("Spiking-StateNet")#encoder_builder("Spiking-STNet")#
        
        
        
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)  # in_channels
        # Support for different versions
        if self.version == 'A':  # Channel concatenation
            n1o, n1i = n1, n1 + ni + no #+ sn1
            n2o, n2i = n2, n2 + n1 + no #+ sn1
            n3o, n3i = n3, n3 + n2 + no #+ sn2
            n4o, n4i = n4, n4 + n3 #+ sn3 #+ en4
            n5o, n5i = n5, n5 + n4 #+ sn4 #+ sn4
        elif self.version == 'B':  # Channel addition
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder
        
        self.tam1 = TAM_module(input_size=1, hidden_size=1, kernel_size=3, activation=None)
        self.tam2 = TAM_module(input_size=1, hidden_size=1, kernel_size=3, activation=None)
        self.tam3 = TAM_module(input_size=1, hidden_size=1, kernel_size=3, activation=None)
        self.tam4 = TAM_module(input_size=1, hidden_size=1, kernel_size=3, activation=None)
        
        self.theta = nn.Conv2d(4, 1, 3, 1, 1)
        self.fi = nn.Conv2d(4, 1, 3, 1, 1)
        self.rho = nn.Conv2d(1, 4, 3, 1, 1)

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2])
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3])
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4])

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

        # Decoder

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)
        
        self.iconv5_spike = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4_spike = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3_spike = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2_spike = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1_spike = Conv2D(n1i, n1, iconv_kernel[4], 1)        
        
        # Decoder for teacher
        '''
        self.sunpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.sunpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.sunpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.sunpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.sunpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.siconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.siconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.siconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.siconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.siconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)
        '''
        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        
        

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)
        
        self.disp4_layer_spike = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer_spike = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer_spike = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer_spike = InvDepth(n1, out_channels=out_channels)        

        # Depth Layers for teacher
        '''
        self.sunpack_disps = nn.PixelShuffle(2)
        self.sunpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.sunpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.sunpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        
        

        self.sdisp4_layer = InvDepth(n4, out_channels=out_channels)
        self.sdisp3_layer = InvDepth(n3, out_channels=out_channels)
        self.sdisp2_layer = InvDepth(n2, out_channels=out_channels)
        self.sdisp1_layer = InvDepth(n1, out_channels=out_channels)
        
        self.sdisp4_layer_spike = InvDepth(n4, out_channels=out_channels)
        self.sdisp3_layer_spike = InvDepth(n3, out_channels=out_channels)
        self.sdisp2_layer_spike = InvDepth(n2, out_channels=out_channels)
        self.sdisp1_layer_spike = InvDepth(n1, out_channels=out_channels)  
        '''
        self.init_weights()
        
        

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def spike_FAM_module(self, x, length):
        """ Frequency aggregation module x.shape=[B,C,H,W] """
        if length == x.size(1):
            return torch.sum(x, dim=1, keepdim=True) / 128.0
        
        else:
            red = x.size(1) - length
            start = random.uniform(0, red)
            spike_window = x[:, start: start + length, :, :]
            
            spike_fre = torch.sum(spike_window, dim=1, keepdim=True)
            
            return spike_fre / 128.0
    
        
    def spike_TAM_module(self, x):
        """ Temporal aggregation module x.shape=[B,C,H,W]"""
        voxels = torch.chunk(x, chunks=4, dim=1)
        
        state1 = None
        voxel_bin = 128 #voxels[0].shape(1)
        for t in range(0, 32):
            y1, state1 = self.tam1(voxels[0][:,t,:,:].unsqueeze(1), state1)
        for t in range(0, 32):
            state2 = state1
            y2, state2 = self.tam2(voxels[1][:,t,:,:].unsqueeze(1), state2)
        for t in range(0, 32):
            state3 = state2
            y3, state3 = self.tam3(voxels[2][:,t,:,:].unsqueeze(1), state3)
        for t in range(0, 32):
            state4 = state3
            y4, state4 = self.tam4(voxels[3][:,t,:,:].unsqueeze(1), state4)
        
        y = torch.stack([y1,y2,y3,y4], 1).squeeze(2)

                    
        return y    

    def downsample(self, x, scale_factor=0.5):
        
        return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    def upsample(self, x, size):
       
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


    def affinity_aggregation(self, tem, fre):
        
        q = self.downsample(self.theta(tem), 0.5)
        k = self.downsample(self.fi(tem), 0.5)
        v = self.downsample(self.rho(fre), 0.5)
        
        
        
        B,C,H,W = q.size()
        _, C_v, _, _ = v.size()
        
        q = q.view(B*C,H,W)
        q = torch.sum(q, dim=1, keepdim=False)
        k = k.view(B*C,H,W)
        k = torch.sum(k, dim=2, keepdim=False)
        v = v.view(B, C_v, H, W)
        
        #print(q.shape, k.shape)
        att = torch.matmul(q.transpose(0,1), k).transpose(0,1)
        att = nn.functional.sigmoid(att).view(B,C,H,W)
        
        
        att = torch.mul(v, att).view(B, C_v, H, W)
        size_u = (int(2*H), int(2*W))
        
        
        return self.upsample(att, size_u)
    
    def forward_spike(self, spike_sequence):
        
        L = int(len(spike_sequence) / 3)
        
        
        current_states = None #spike_sequence
        current_lstm = None
        
        spike_t = self.spike_TAM_module(spike_sequence)
        spike_f = self.spike_FAM_module(spike_sequence, 128)
        
        st_feature = self.affinity_aggregation(spike_t, spike_f)
        prime_feature = st_feature + spike_t 
        
        spike_feat = self.spike_encoder.forward_events(prime_feature)
        
        current_states = spike_feat
        snn_features = spike_feat

        
        return current_states, snn_features
           
        
    
    def forward_rgb(self, rgb):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        rgb_shape = rgb.shape
        
        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips
        #print(x1p.shape, x2p.shape, x3p.shape, x4p.shape, x5p.shape)

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        return [x, x1p, x2p, x3p, x4p, x5p, rgb_shape]
        

    def forward_decoder(self, fusion_codings):
        prec, x1p, x2p, x3p, x4p, x5p = fusion_codings
        #print(x1p.shape, x2p.shape, x3p.shape, x4p.shape, x5p.shape)

        skip1 = prec # 64
        skip2 = x1p # 64
        skip3 = x2p # 64
        skip4 = x3p # 128
        skip5 = x4p # 256
        
        # Decoder

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        if self.training:
            return {
                'inv_depths': [disp1, disp2, disp3, disp4],
            }
        else:
            return {
                'inv_depths': disp1,
            }             

    def forward_teacher_decoder(self, fusion_codings):
        prec, x1p, x2p, x3p, x4p, x5p = fusion_codings
        

        skip1 = prec # 64
        skip2 = x1p # 64
        skip3 = x2p # 64
        skip4 = x3p # 128
        skip5 = x4p # 256
        
        # Decoder

        unpack5 = self.sunpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.siconv5(concat5)

        unpack4 = self.sunpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = sunpack4 + skip4
        iconv4 = self.siconv4(concat4)
        disp4 = self.sdisp4_layer(iconv4)
        udisp4 = self.sunpack_disp4(disp4)

        unpack3 = self.sunpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.siconv3(concat3)
        disp3 = self.sdisp3_layer(iconv3)
        udisp3 = self.sunpack_disp3(disp3)

        unpack2 = self.sunpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.siconv2(concat2)
        disp2 = self.sdisp2_layer(iconv2)
        udisp2 = self.sunpack_disp2(disp2)

        unpack1 = self.sunpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.siconv1(concat1)
        disp1 = self.sdisp1_layer(iconv1)

        if self.training:
            return {
                'inv_depths': [disp1, disp2, disp3, disp4],
            }
        else:
            return {
                'inv_depths': disp1,
            }   
        
    def forward(self, rgb, spike, epoch):
        encodings = {}
        if True:        
            rgb_feat_list = self.forward_rgb(rgb)
            [x, x1p, x2p, x3p, x4p, x5p, rgb_shape] = rgb_feat_list
            
            y, spike_features = self.forward_spike(spike)
            [xs, x1s, x2s, x3s, x4s, x5s] = spike_features
            
            x = x + xs
            x1p = x1p + x1s
            x2p = x2p + x2s
            x3p = x3p + x3s
            x4p = x4p + x4s
            x5p = x5p + x5s          
                             
            
            if self.version_II == 'B':

                x = (1-0.1 * (epoch+1)) * x + xs 
                x1p = (1-0.1 * (epoch+1)) * x1p + x1s 
                x2p = (1-0.1 * (epoch+1)) * x2p + x2s 
                x3p = (1-0.1 * (epoch+1)) * x3p + x3s 
                x4p = (1-0.1 * (epoch+1)) * x4p + x4s 
                x5p = (1-0.1 * (epoch+1)) * x5p + x5s              
               
            #decodings = self.forward_teacher_decoder((x, x1p, x2p, x3p, x4p, x5p))
            decodings = self.forward_decoder((x, x1p, x2p, x3p, x4p, x5p))
            encodings['spike_features'] = [xs, x1s, x2s, x3s, x4s, x5s]
            encodings['fusion_features'] = [x, x1p, x2p, x3p, x4p, x5p]
        
        return decodings, encodings
            
    def pass_spike_features(self, spike):
        y, spike_features = self.forward_spike(spike) 
        return spike_features
    
    #def forward_single_branch(self, rgb, spike, connector):
    def forward_single_branch(self, spike):
        encodings = {}
        
        if self.forward_single:    
            pass
        else:                
            y, spike_features = self.forward_spike(spike)
            [xs, x1s, x2s, x3s, x4s, x5s] = spike_features
            [xc, x1c, x2c, x3c, x4c, x5c] = connector
            
            if self.version_III == 'D':
                x = xs + xc
                x1p = x1s + x1c
                x2p = x2s + x2c
                x3p = x3s + x3c
                x4p = x4s + x4c
                x5p = x5s + x5c             
            decodings = self.forward_decoder((x, x1p, x2p, x3p, x4p, x5p))

        y, spike_features = self.forward_spike(spike)
        [x, x1p, x2p, x3p, x4p, x5p] = spike_features
        [x, x1p, x2p, x3p, x4p, x5p, rgb_shape] = self.forward_rgb(spike)
        
        decodings = self.forward_decoder((x, x1p, x2p, x3p, x4p, x5p))
        spike_features = [x, x1p, x2p, x3p, x4p, x5p]
        
        return decodings, spike_features   
        
          