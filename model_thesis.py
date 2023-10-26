"""
Created on 2023/09/20 15:29
@author: Hailong Lin
@source: https://github.com/Zhenzi-Weng/DeepSC-S.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from channel import Channel
from config import Config
from speech_processing import wav_norm, wav_denorm, enframe, deframe
################## CONFIG ##################
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997
DEPTH = 16
CARDINALITY = 2
REDUCTION_RATIO = 4
BIAS = False

norm_layers = [nn.BatchNorm2d, nn.Identity]
norm_layer = norm_layers[1]
class ConvBnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=5):
        super(ConvBnLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=BIAS)
        self.batch_norm = norm_layer(out_channels)
    
    def forward(self, x):
        output = self.conv(x)
        output = self.batch_norm(output)
        return output

class ConvtransBnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, out_padding=0, kernel_size=5):
        super(ConvtransBnLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        # TODO: out_padding
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, 
                                            kernel_size=kernel_size,
                                            stride=stride, padding=padding, 
                                            output_padding=out_padding, bias=BIAS)
        self.batch_norm = norm_layer(out_channels)
        
    def forward(self, x):
        output = self.convtrans(x)
        output = self.batch_norm(output)
        return output

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.net = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        output = self.net(x)
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        return output

class TransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(TransformLayer, self).__init__()
        self.conv_bn = ConvBnLayer(in_channels, out_channels, stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        
        output = self.conv_bn(x)
        output = self.relu(output)
        return output

class SplitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SplitLayer, self).__init__()
        self.layers_split = nn.ModuleList()
        for _ in range(CARDINALITY):
            split = TransformLayer(in_channels, out_channels, stride)
            self.layers_split.append(split)
    def forward(self, x):
        output_list = [layer(x) for layer in self.layers_split]
        output = torch.cat(output_list, dim=1)
        return output

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride,
                              bias=BIAS)
        # self.bn = nn.LazyBatchNorm2d(eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        return output

class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(SELayer, self).__init__()
        self.net = nn.Sequential(
            GlobalAveragePooling(),
            nn.Linear(in_channels, int(out_channels / reduction_ratio), bias=BIAS),
            nn.ReLU(),
            nn.Linear(int(out_channels / reduction_ratio), out_channels, bias=BIAS),
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.net(x)
        shape = output.shape
        output = output.reshape(shape[0], shape[1], 1, 1)
        return output
    
class SEResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEResNet, self).__init__()
        self.split_layer = SplitLayer(in_channels, DEPTH, stride=(1, 1))
        self.transition_layer = TransitionLayer(DEPTH*CARDINALITY, out_channels)
        self.se_layer = SELayer(out_channels, out_channels, reduction_ratio=REDUCTION_RATIO)
    def forward(self, x):
        output = self.split_layer(x)
        transition_output = self.transition_layer(output)
        se_output = self.se_layer(transition_output)
        output = x + se_output * transition_output
        return output
################## model ##################
# Semantic Encoder
class SemEncoder(nn.Module):
    def __init__(self, args, in_channels=1):
        super(SemEncoder, self).__init__()
        self.num_frame = args.num_frame
        self.frame_length = int(args.sr * args.frame_size)
        self.stride_length = int(args.sr * args.stride_size)

        self.sem_enc_outdims = args.sem_enc_outdims
        self.conv = nn.Sequential(
            ConvBnLayer(in_channels, self.sem_enc_outdims[0], stride=(1, 1)),
            nn.ReLU(),
            ConvBnLayer(self.sem_enc_outdims[0], self.sem_enc_outdims[1], stride=(1, 1)),
            nn.ReLU()
        )
        se_resnets = nn.ModuleList()
        # iidx = 0
        for idx, outdim in enumerate(self.sem_enc_outdims[2:], 2):
            se_resnet = SEResNet(self.sem_enc_outdims[idx-1], outdim)
            se_resnets.append(se_resnet)
            se_resnets.append(nn.ReLU())
            # iidx += 1
        self.se_resnets_net = nn.Sequential(*se_resnets)
        # print("idx:", iidx)
    def forward(self, _input):
        _input, batch_mean, batch_var = wav_norm(_input)
        _input = enframe(_input, self.num_frame, self.frame_length, self.stride_length)
        _input = _input.unsqueeze(1)
        output = self.conv(_input)
        output= self.se_resnets_net(output)
        return output, batch_mean, batch_var

    
# Channel Encoder
class ChanEncoder(nn.Module):
    def __init__(self, in_channels, args):
        super(ChanEncoder, self).__init__()
        self.chan_enc_filters = args.chan_enc_filters
        self.conv_bn = ConvBnLayer(in_channels, self.chan_enc_filters[0],
                                   stride=(1, 1))

    def forward(self, x):
        output = self.conv_bn(x)
        return output

def chan_layer(_input, snr, channel_type="AWGN", P=2):
    batch_size = _input.size(0)
    x = _input.view(batch_size, -1)
    _output = Channel(x, snr, channel_type=channel_type, P=P)
    _output = _output.reshape(_input.shape)
    return _output

class ChanDecoder(nn.Module):
    def __init__(self, in_channels, args):
        super(ChanDecoder, self).__init__()
        self.chan_dec_filters = args.chan_dec_filters
        self.conv_bn = ConvBnLayer(in_channels, self.chan_dec_filters[0],
                                   stride=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv_bn(x)
        output = self.relu(output)
        return output

class SemDecoder(nn.Module):
    def __init__(self, in_channels, args, last_channels=1, input_size=(3, 64, 64)):
        super(SemDecoder, self).__init__()
        self.num_frame = args.num_frame
        self.frame_length = int(args.sr * args.frame_size)
        self.stride_length = int(args.sr * args.stride_size)

        self.sem_dec_outdims = args.sem_dec_outdims
        se_resnets = nn.ModuleList()
        output_padding_x = [0, 0]
        output_padding_y = [0, 0]
        # if input_size[1] % 4 == 0:
        #     output_padding_x[0] = 1
        # if input_size[1] % 2 == 0:
        #     output_padding_x[1] = 1
        # if input_size[2] % 4 == 0:
        #     output_padding_y[0] = 1            
        # if input_size[2] % 2 == 0:
        #     output_padding_y[1] = 1
        for idx, outdim in enumerate(self.sem_dec_outdims[:-2]):
            if idx == 0:
                in_chan = in_channels
            else:
                in_chan = self.sem_dec_outdims[idx - 1]
            
            se_resnet = SEResNet(in_chan, outdim)
            se_resnets.append(se_resnet)
            se_resnets.append(nn.ReLU())
        self.se_resnets_net = nn.Sequential(*se_resnets)
        self.convtrans = nn.Sequential(
            ConvtransBnLayer(self.sem_dec_outdims[-3], self.sem_dec_outdims[-2], stride=(1, 1),
                             out_padding=(output_padding_x[0], output_padding_y[0])),
            nn.ReLU(),
            # ConvtransBnLayer(self.sem_dec_outdims[-2], self.sem_dec_outdims[-1], stride=(1, 1),
            #                  out_padding=(output_padding_x[1], output_padding_y[1])),
            # nn.ReLU()
            )
        # last Layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(self.sem_dec_outdims[-1], last_channels,
                      stride=(1, 1), kernel_size=(1, 1), bias=False),
            )
    def forward(self, x, batch_mean=None, batch_var=None):
        # output = x
        output = self.se_resnets_net(x)
        # output = self.convtrans(output)
        output = self.final_layer(output)
        output = output.squeeze(1)
        output = deframe(output, self.frame_length, self.stride_length)
        output = wav_denorm(output, batch_mean, batch_var)
        return output


class Transmitter(nn.Module):
    def __init__(self, in_channels, args):
        super(Transmitter, self).__init__()
        self.sem_encoder = SemEncoder(in_channels=in_channels, args=args)
        self.chan_encoder = ChanEncoder(in_channels=args.sem_enc_outdims[-1], args=args)
    
    def forward(self, x):
        output, batch_mean, batch_var = self.sem_encoder(x)
        output = self.chan_encoder(output)
        return output, batch_mean, batch_var

class Receiver(nn.Module):
    def __init__(self, args, last_channels, shape):
        super(Receiver, self).__init__()
        self.chan_decoder = ChanDecoder(in_channels=args.chan_enc_filters[-1], args=args)
        self.sem_decoder = SemDecoder(in_channels=args.chan_dec_filters[-1], args=args, 
                                      last_channels=last_channels, input_size=shape)
    def forward(self, x, batch_mean=None, batch_var=None):
        output = self.chan_decoder(x)
        output = self.sem_decoder(output, batch_mean, batch_var)
        return output

class DeepSCThesis(nn.Module):
    def __init__(self, args, shape=(1, 128, 128), in_channels=1, config=None):
        super(DeepSCThesis, self).__init__()
        self.transmitter = Transmitter(in_channels, args)
        self.receiver = Receiver(args, in_channels, shape)
        self.channel = Channel(config)
        self.initialize_weights()
    def forward(self, x, snr=None):
        _output, batch_mean, batch_var = self.transmitter(x)
        # with torch.no_grad():
        _output, channel_usage = self.channel(_output, param=snr)
        _output = self.receiver(_output, batch_mean, batch_var)
        return _output, channel_usage
    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

def test():
    from torchsummary import summary
    from utils import parse_args
    import numpy as np
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channel = 1
    last_channels = 1
    batch_size = 64
    shape = (3, 32, 32)
    deepsc = DeepSCThesis(args).to(device)
    import time
    deepsc.initialize_weights()
    time.sleep(10)
    # x = torch.randn((1, 128*128))
    # y = deepsc(x)
    # print(y.shape)
    # transmitter = Transmitter(in_channel, args).to(device)
    # transmitter = Transmitter(in_channel, args)
    # receiver = Receiver(args, last_channels, shape=(1, 128, 128))
    # # print(receiver(x).shape)
    # se_res = SEResNet(32, 32)
    # # # summary(transmitter, (1, 128, 128))
    from torchstat import stat 
    # stat(deepsc, (1, 128*128))
    # # stat(se_res, (32, 128, 128))
    # # stat(transmitter, (1, 128, 128))
    # stat(receiver, (8, 128, 128))
    # # # from ptflops import get_model_complexity_info
    # # from thop import profile
    # # input = torch.randn(1, 32, 128, 128)
    # # flops, params = profile(se_res, inputs=(input, ))
    # # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # # print('Params = ' + str(params/1000**2) + 'M')
    # # x = nn.Conv2d(32, 32, kernel_size=(1, 1))
    # # print("x:", x.flops())
    # # stat(x, (32, 128, 128))
    # # trans = TransitionLayer(32, 32)
    # # stat(trans, (32, 128, 128))
    # # # macs, params = get_model_complexity_info(se_res, (32, 128, 128), as_strings=True,
    # # #                                     print_per_layer_stat=True, verbose=True)
    # # # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # # # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # # # stat(transmitter, (1, 128, 128))
if __name__=="__main__":
    test()