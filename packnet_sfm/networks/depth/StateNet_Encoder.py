import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from packnet_sfm.networks.depth.submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, Recurrent2ConvLayer, RecurrentPhasedConvLayer, ResidualBlock, ConvLSTM, \
    ConvGRU, RecurrentResidualLayer

from packnet_sfm.networks.depth.submodules_spiking import \
    ConvLIF, ConvPLIF, ConvALIF, ConvXLIF, ConvLIFRecurrent, ConvPLIFRecurrent, ConvALIFRecurrent, \
    ConvXLIFRecurrent, SpikingResidualBlock, SpikingRecurrentConvLayer, SpikingUpsampleConvLayer, \
    SpikingTransposedConvLayer

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


def state_sum(x1, x2, conv=None, lstm_states=None):
    return x1.add(x2)


def state_conv(x1, x2, conv, lstm_states=None):
    return conv(torch.cat([x1, x2], dim=1))


def state_conv_lstm(x1, state, conv_lstm, lstm_states):
    return conv_lstm(x1, lstm_states)


def state_conv_gru(x1, state, conv_gru, lstm_states=None):
    return conv_gru(x1, state)

def state_spiking_conv_gru(x1, state, spiking_conv_gru, lstm_states=None):
    return spiking_conv_gru(x1, state)
    
    
class BaseStateNet(nn.Module):
    def __init__(self, num_input_channels_rgb, num_input_channels_events, num_output_channels=1, skip_type='sum',
                 state_combination='sum', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True,
                 recurrent_block_type='convlstm', baseline=False):

        super(BaseStateNet, self).__init__()

        self.num_input_channels_rgb = num_input_channels_rgb
        self.num_input_channels_events = num_input_channels_events
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')
        self.state_combination = state_combination
        if self.state_combination == 'sum':
            self.apply_state_combination = state_sum
        elif self.state_combination == 'conv':
            self.apply_state_combination = state_conv
        elif self.state_combination == 'convlstm':
            self.apply_state_combination = state_conv_lstm
        elif self.state_combination == 'convgru':
            self.apply_state_combination = state_conv_gru
        elif self.state_combination == 'spiking-convgru':
            self.apply_state_combination = state_spiking_conv_gru        
        else:
            raise KeyError('Could not identify state_combination, please add "state_combination":'
                           ' "sum", "conv", "convlstm" or "convgru" to config["model"]')
        self.recurrent_block_type = recurrent_block_type
        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)
        self.norm = norm
        # self.kernel_size = kernel_size
        self.baseline = baseline

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert (self.num_input_channels_rgb > 0 or self.num_input_channels_events > 0)
        assert (self.num_output_channels > 0)

        self.encoder_input_sizes = []
        #for i in range(self.num_encoders):
        #    self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))
            
        self.encoder_input_sizes = [64, 64, 64, 64, 128, 256]
        self.encoder_output_sizes = [64, 64, 64, 128, 256, 512]
        self.encoder_stride = [1, 2, 2, 2, 2, 2]
        
        #self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)



class StateNetPhasedRecurrent(BaseStateNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels_rgb, num_input_channels_events, num_output_channels=1, skip_type='sum',
                 state_combination='sum', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True, recurrent_block_type='convlstm',
                 baseline=False):

        super(StateNetPhasedRecurrent, self).__init__(num_input_channels_rgb, num_input_channels_events,
                                                      num_output_channels, skip_type, state_combination, activation,
                                                      num_encoders, base_num_channels, num_residual_blocks, norm,
                                                      use_upsample_conv, recurrent_block_type, baseline)
        
        
        
        self.baseline = baseline

        #self.head_rgb = ConvLayer(self.num_input_channels_rgb, self.base_num_channels,
        #                          kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W
        #self.encoders_rgb = nn.ModuleList()

        if not bool(self.baseline):
            self.head_events = ConvLayer(self.num_input_channels_events, self.base_num_channels,
                                         kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W
            self.encoders_events = nn.ModuleList()

        if self.state_combination == 'sum':
            self.state_combination_events = []
            self.state_combination_images = []
        elif self.state_combination == 'convlstm' or self.state_combination == 'conv' \
                or self.state_combination == 'convgru':
            if not bool(self.baseline):
                self.state_combination_events = nn.ModuleList()
            

        for input_size, output_size, encoder_stride in zip(self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_stride):
            if self.recurrent_block_type == 'convlstm':
                
                if encoder_stride == 1:
                    self.encoders_events.append(Recurrent2ConvLayer(input_size, output_size,
                                                                   kernel_size=3, stride=1, padding=1,
                                                                   norm=self.norm,
                                                                   recurrent_block_type=self.recurrent_block_type))
                
                else:
                    self.encoders_events.append(Recurrent2ConvLayer(input_size, output_size,
                                                                   kernel_size=5, stride=2, padding=2,
                                                                   norm=self.norm,
                                                                   recurrent_block_type=self.recurrent_block_type))
            elif self.recurrent_block_type == 'conv':
                #self.encoders_rgb.append(ConvLayer(input_size, output_size, kernel_size=5,
                                                   #stride=2, padding=2, norm=self.norm))
                
                
                if not bool(self.baseline):
                    self.encoders_events.append(ConvLayer(input_size, output_size, kernel_size=5,
                                                          stride=2, padding=2, norm=self.norm))

            if self.state_combination == 'sum':
                if not bool(self.baseline):
                    self.state_combination_events.append(None)


            elif self.state_combination == 'convlstm' or self.state_combination == "convgru":
                if not bool(self.baseline):
                    self.state_combination_events.append(RecurrentConvLayer(output_size, output_size,
                                                                            kernel_size=5, stride=1, padding=2,
                                                                            norm=self.norm,
                                                                            recurrent_block_type=self.state_combination))


            elif self.state_combination == 'conv':
                if not bool(self.baseline):
                    self.state_combination_events.append(ConvLayer(output_size * 2, output_size,
                                                                   kernel_size=5, stride=1, padding=2,
                                                                   norm=self.norm))
                self.state_combination_images.append(ConvLayer(output_size * 2, output_size,
                                                               kernel_size=5, stride=1, padding=2,
                                                               norm=self.norm))

        self.build_resblocks()
        #self.build_decoders()
        #self.build_prediction_layer()

    def forward_events(self, x, prev_super_states, prev_states_lstm):
        
        x = self.head_events(x)
        
        if prev_super_states is None:
            # print("state is none, state is initialized")
            prev_super_states = []
            B, C, H, W = 1, 3, 192, 640 #item['image'].shape
            for i in range(self.num_encoders):
                H_state = int(H / pow(2, i + 1))
                W_state = int(W / pow(2, i + 1))
                N_channels = int(self.base_num_channels * pow(2, i + 1))
                if not bool(self.baseline) and self.state_combination == 'convlstm':
                    # for state with convlstm, state has two entries (hidden & cell state)
                    prev_super_states.append([torch.zeros([B, N_channels, H_state, W_state]).to(x.device),
                                              torch.zeros([B, N_channels, H_state, W_state]).to(x.device)])
                else:
                    prev_super_states.append(torch.zeros([B, N_channels, H_state, W_state]).to(x.device))        

        if prev_states_lstm is None:
            prev_states_lstm = {}
            prev_states_lstm['encoders'] = [None] * self.num_encoders
            prev_states_lstm['state_comb'] = [None] * self.num_encoders

        super_states = []
        states_lstm = {'encoders': [], 'state_comb': []}

        for i, encoder in enumerate(self.encoders_events):
            if self.recurrent_block_type == 'conv':
                x = encoder(x)
                state_lstm_encoder = None
            elif self.recurrent_block_type == 'convlstm':
                x, state_lstm_encoder = encoder(x, prev_states_lstm['encoders'][i])

            if self.state_combination == "convlstm":  # and prev_states_lstm['state_comb'][i] is not None:
                # for statenet with lstm: cell state is from last run_through, specific to the event encoder.
                # Hidden state is the last superstate, independent of whether it was generated by the event or image encoder.
                _, super_state = self.apply_state_combination(x, prev_super_state[i],
                                                              self.state_combination_events[i],
                                                              prev_super_state[i])

                state_lstm_state_comb = super_state
            else:
                super_state, state_lstm_state_comb = self.apply_state_combination(x, prev_super_states[i],
                                                                                  self.state_combination_events[i],
                                                                                  prev_states_lstm['state_comb'][i])
            # assert torch.all(super_state.eq(state_lstm_state_comb))
            super_states.append(super_state)
            states_lstm['encoders'].append(state_lstm_encoder)
            states_lstm['state_comb'].append(state_lstm_state_comb)
            print(len(super_states))

        return super_states, states_lstm




class SpikingStateNetPhasedRecurrent(BaseStateNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels_rgb, num_input_channels_events, num_output_channels=1, skip_type='sum',
                 state_combination='sum', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True, recurrent_block_type='convlstm',
                 baseline=False):

        super(SpikingStateNetPhasedRecurrent, self).__init__(num_input_channels_rgb, num_input_channels_events,
                                                      num_output_channels, skip_type, state_combination, activation,
                                                      num_encoders, base_num_channels, num_residual_blocks, norm,
                                                      use_upsample_conv, recurrent_block_type, baseline)
        
        
        
        self.baseline = baseline

        self.head_rgb = ConvLayer(self.num_input_channels_rgb, self.base_num_channels,
                                  kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W
        self.encoders_rgb = nn.ModuleList()

        if not bool(self.baseline):
            self.head_events = ConvLayer(self.num_input_channels_events, self.base_num_channels,
                                         kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W
            self.encoders_events = nn.ModuleList()

        if self.state_combination == 'sum':
            self.state_combination_events = []
            self.state_combination_images = []
        elif self.state_combination == 'convlstm' or self.state_combination == 'conv' \
                or self.state_combination == 'convgru' or self.state_combination == 'spiking-convgru':
            if not bool(self.baseline):
                self.state_combination_events = nn.ModuleList()
            self.state_combination_images = nn.ModuleList()

        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            if self.recurrent_block_type == 'convlstm':
                self.encoders_rgb.append(Recurrent2ConvLayer(input_size, output_size,
                                                            kernel_size=5, stride=2, padding=2,
                                                            norm=self.norm,
                                                            recurrent_block_type=self.recurrent_block_type))
                if not bool(self.baseline):
                    self.encoders_events.append(Recurrent2ConvLayer(input_size, output_size,
                                                                   kernel_size=5, stride=2, padding=2,
                                                                   norm=self.norm,
                                                                   recurrent_block_type=self.recurrent_block_type))
            elif self.recurrent_block_type == 'conv':

                if not bool(self.baseline):
                    #self.encoders_events.append(ConvLayer(input_size, output_size, kernel_size=5,
                    #                                      stride=2, padding=2, norm=self.norm))
                    self.encoders_events.append(ConvLayer(input_size, output_size, kernel_size=5,
                                                          stride=2, padding=2, norm=self.norm))
                    
            if self.state_combination == 'sum':
                if not bool(self.baseline):
                    self.state_combination_events.append(None)
                self.state_combination_images.append(None)

            elif self.state_combination == 'convlstm' or self.state_combination == "convgru":
                if not bool(self.baseline):
                    self.state_combination_events.append(RecurrentConvLayer(output_size, output_size,
                                                                            kernel_size=5, stride=1, padding=2,
                                                                            norm=self.norm,
                                                                            recurrent_block_type=self.state_combination))
                self.state_combination_images.append(RecurrentConvLayer(output_size, output_size,
                                                                        kernel_size=5, stride=1, padding=2,
                                                                        norm=self.norm,
                                                                        recurrent_block_type=self.state_combination))
            
            elif self.state_combination == 'spiking-convlstm' or self.state_combination == "spiking-convgru":
                if not bool(self.baseline):
                    self.state_combination_events.append(SpikingRecurrentConvLayer(output_size, output_size,
                                                                            kernel_size=5, stride=1, 
                                                                            recurrent_block_type="lif",
                                                                            activation_ff="arctanspike",
                                                                            activation_rec="arctanspike"))            
            
            elif self.state_combination == 'conv':
                if not bool(self.baseline):
                    self.state_combination_events.append(ConvLayer(output_size * 2, output_size,
                                                                   kernel_size=5, stride=1, padding=2,
                                                                   norm=self.norm))



        self.build_prediction_layer()

    def forward_events(self, x, prev_super_states, prev_states_lstm, times, rgb_shape):
        x = self.head_events(x)
        snn_feature_list = list()
        
        
        if prev_super_states is None:
            # print("state is none, state is initialized")
            prev_super_states = []
            B, C, H, W = rgb_shape #item['image'].shape
            for i in range(self.num_encoders):
                H_state = int(H / pow(2, i + 1))
                W_state = int(W / pow(2, i + 1))
                N_channels = int(self.base_num_channels * pow(2, i + 1))
                if not bool(self.baseline) and self.state_combination == 'convlstm':
                    # for state with convlstm, state has two entries (hidden & cell state)
                    prev_super_states.append([torch.zeros([B, N_channels, H_state, W_state]).to(x.device),
                                              torch.zeros([B, N_channels, H_state, W_state]).to(x.device)])
                else:
                    prev_super_states.append(torch.zeros([B, N_channels, H_state, W_state]).to(x.device))        
        
        if prev_states_lstm is None:
            prev_states_lstm = {}
            prev_states_lstm['encoders'] = [None] * self.num_encoders
            prev_states_lstm['state_comb'] = [None] * self.num_encoders
        
        super_states = []
        prev_super_states = [None, None, None, None, None]
        states_lstm = {'encoders': [], 'state_comb': []}
        #super_states = [{}] * num_recurrent_units
        
        for i, encoder in enumerate(self.encoders_events):
            if self.recurrent_block_type == 'conv':
                x = encoder(x)
                snn_feature_list.append(x)
                
                state_lstm_encoder = None


            if self.state_combination == "convlstm":  # and prev_states_lstm['state_comb'][i] is not None:
                # for statenet with lstm: cell state is from last run_through, specific to the event encoder.
                # Hidden state is the last superstate, independent of whether it was generated by the event or image encoder.
                _, super_state = self.apply_state_combination(x, prev_super_state[i],
                                                              self.state_combination_events[i],
                                                              prev_super_state[i])

                state_lstm_state_comb = super_state

            elif self.state_combination == "spiking-convgru":
                #prev_super_states[0] = None
                #print(prev_super_states[i])
                #print(i)
                super_state, state_lstm_state_comb = self.apply_state_combination(x, prev_super_states[i],
                                                                                  self.state_combination_events[i],
                                                                                  prev_states_lstm['state_comb'][i])
                                                                                  
                prev_super_states[i] = (state_lstm_state_comb[0,...].squeeze(0), state_lstm_state_comb[1,...].squeeze(1))                                                                
                 
                print(state_lstm_state_comb)
                
            else:
                prev_super_states[0] = None
                super_state, state_lstm_state_comb = self.apply_state_combination(x, prev_super_states[i],
                                                                                  self.state_combination_events[i],
                                                                                  prev_states_lstm['state_comb'][i])
                
            # assert torch.all(super_state.eq(state_lstm_state_comb))
            super_states.append(super_state)
            #super_states.append((state_lstm_state_comb[0,...].squeeze(0), state_lstm_state_comb[1,...].squeeze(1)))
            states_lstm['encoders'].append(state_lstm_encoder)
            states_lstm['state_comb'].append(state_lstm_state_comb)

        return super_states, states_lstm, snn_feature_list
        
        
        
        