# -*- coding: utf-8 -*-
"""
Created on 2023/10/12
Author: Hailong Lin
File: speech_processing.py
Email: linhl@emnets.org
Last modified: 2023/10/12 17:51
"""
import torch

def wav_norm(wav_input):
    batch_var, batch_mean = torch.var_mean(wav_input, dim=-1, keepdim=True)
    wav_input_norm = (wav_input - batch_mean) / torch.sqrt(batch_var)
    return wav_input_norm, batch_mean, batch_var

def wav_denorm(wav_output, batch_mean, batch_var):
    wav_output_denorm = wav_output * torch.sqrt(batch_var) + batch_mean
    return wav_output_denorm

def enframe(wav_input, num_frame, frame_length, stride_length):
    batch_size, num_samples = wav_input.size()
    
    # 计算输出帧的总数
    num_output_frames = (num_samples - frame_length) // stride_length + 1
    
    # 初始化一个空的帧张量
    frame_input = torch.empty(batch_size, num_frame, frame_length, dtype=wav_input.dtype, device=wav_input.device)
    
    for i in range(num_frame):
        # 计算每帧的起始位置
        start = i * stride_length
        
        # 计算每帧的结束位置
        end = start + frame_length
        
        # 从输入张量中提取每帧数据
        frame_input[:, i, :] = wav_input[:, start:end]
    
    return frame_input

def deframe(frame_output, frame_length, stride_length):
    batch_size, num_frame, _ = frame_output.size()
    # 计算原始音频数据的总长度
    num_samples = (num_frame - 1) * stride_length + frame_length
    # 初始化一个空的音频张量
    wav_output = torch.empty(batch_size, num_samples, dtype=frame_output.dtype, device=frame_output.device)
    for i in range(num_frame):
        # 计算每帧的起始位置
        start = i * stride_length
        # 计算每帧的结束位置
        end = start + frame_length
        # 将每帧的数据放回到原始音频张量中
        wav_output[:, start:end] = frame_output[:, i, :]
    return wav_output

if __name__=="__main__":
    num_frame = 128
    frame_length = int(8000*0.016)
    stride_length = int(8000*0.016)
    x = torch.rand((32, 16384))
    _input, batch_mean, batch_var = wav_norm(x)
    # print(_input.shape, batch_mean.shape, batch_var.shape, batch_mean[0], torch.mean(x, dim=-1)[0])
    __input = enframe(_input, num_frame, frame_length, stride_length)
    print(__input.shape)
    _output = deframe(__input, frame_length, stride_length)
    print(_output.shape)
    for i in range(128*128):
        for j in range(_output.shape[0]):
            if _input[j, i] != _output[j, i]:
                raise ValueError("NO")
    
