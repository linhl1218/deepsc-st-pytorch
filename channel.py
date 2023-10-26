import torch.nn as nn
import numpy as np
import os
import torch

class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']
        # self.device = config.device
        # print(config)
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))
        if self.chan_type == 3 or self.chan_type == 'rician':
            self.rician_k = config.channel["rician_k"]
        else:
            self.rician_k = None

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        if device == -1:
            device = torch.device("cpu")
        # print("device:", device)
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        if device == -1:
            device = torch.device("cpu")

        h_real = torch.randn(np.shape(input_layer), device=device)
        h_imag = torch.randn(np.shape(input_layer), device=device)
        h = h_real + 1j * h_imag
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        y_com = input_layer * h + noise
        y = y_com / h
        return y
    
    def rician_noise_layer(self, input_layer, std):
        k = self.rician_k
        device = input_layer.get_device()
        if device == -1:
            device = torch.device("cpu")
        h_real = torch.randn(np.shape(input_layer), device=device)
        h_imag = torch.randn(np.shape(input_layer), device=device)
        h_rayleigh = h_real + 1j * h_imag
        h = torch.sqrt(torch.tensor(k / (k + 1))) + torch.sqrt(torch.tensor(1 / (k + 1))) * h_rayleigh / torch.sqrt(torch.tensor(2))
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        y_com = input_layer * h + noise
        y = y_com / h
        return y
    
    def forward(self, input, param=None, avg_pwr=None, power=1):
        if param is not None:
            self.chan_param = param
        if avg_pwr is None:
            avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        channel_in = channel_in[::2] + channel_in[1::2] * 1j
        channel_usage = channel_in.numel()
        channel_output = self.channel_forward(channel_in)
        channel_rx = torch.zeros_like(channel_tx.reshape(-1))
        channel_rx[::2] = torch.real(channel_output)
        channel_rx[1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output
        
        elif self.chan_type == 3 or self.chan_type == 'rician':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.rician_noise_layer(channel_tx, std=sigma)
            return chan_output