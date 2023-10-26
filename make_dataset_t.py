# -*- coding: utf-8 -*-
"""
Created on 2023/10/12
Author: Hailong Lin
File: make_dataset.py
Email: linhl@emnets.org
Last modified: 2023/10/12 16:39
"""
import os
import random
import argparse
import timeit
import sys
import torchaudio
import torch
import numpy as np
from scipy.io import wavfile
from utils import create_dir, resampler

###############    define global parameters    ###############
def parse_args():
    parser = argparse.ArgumentParser(description="Convert the set of .wavs to .TFRecords")
    
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frame in each barch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride") 

    parser.add_argument("--wav_path", type=str, default="./dataset/raw_data/noisy_trainset_28spk_wav/",
                        help="path of wavset")
    parser.add_argument("--save_path", type=str, default="./dataset/deepsc/",
                        help="path to save .pth file")
    parser.add_argument("--valid_percent", type=float, default=0.05, help="percent of validset in total dataset")
    parser.add_argument("--trainset_filename", type=str, default="trainset_t1.pth", help=".tfrecords filename of trainset")
    parser.add_argument("--validset_filename", type=str, default="validset_t1.pth", help=".tfrecords filename of validset")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:", args)
frame_length = int(args.sr*args.frame_size)
stride_length = int(args.sr*args.stride_size)
window_size = args.num_frame*stride_length+frame_length-stride_length
batch_size = 32
num_gpu = 1
global_batch_size = batch_size*num_gpu

def wav_processing(wav_file, window_size, dataset):
    wav_samples, sr = torchaudio.load(wav_file)
    if sr != args.sr:
        wav_samples = resampler(sr, wav_samples, args.sr)
    wav_samples = wav_samples.squeeze(0)
    num_samples = wav_samples.shape[0]
    # dataset = []
    if num_samples > window_size:
        num_slices = num_samples // window_size + 1
        wav_samples = torch.cat((wav_samples, wav_samples), dim=0)
        wav_samples = wav_samples[0:window_size * num_slices]
        wav_slices = wav_samples.reshape(-1, window_size)
        for wav_slice in wav_slices:
            if torch.mean(torch.abs(wav_slice) / 2) < 0.015:
                num_slices -= 1
            else:
                dataset.append(wav_slice.unsqueeze(0))
    else:
        num_slices = 1
        while wav_samples.shape[0] < window_size:
            wav_samples = torch.cat((wav_samples, wav_samples), dim=0)
        wav_slice = wav_samples[0:window_size]
        if torch.mean(torch.abs(wav_slice) / 2) < 0.015:
            num_slices -= 1
        else:
            dataset.append(wav_slice.unsqueeze(0))
    return num_slices, dataset



if __name__=="__main__":
    ###########################################################################
    wav_files = [os.path.join(args.wav_path, wav) for wav in os.listdir(args.wav_path) if wav.endswith(".wav")]
    num_wav_files = len(wav_files)
    random.shuffle(wav_files)
    num_validset_wav_files = int(args.valid_percent * num_wav_files)
    num_trainset_wav_files = num_wav_files - num_validset_wav_files
    trainset_wav_files = wav_files[0:num_trainset_wav_files]
    validset_wav_files = wav_files[num_trainset_wav_files:num_wav_files]

    num_trainset_wav_files = len(trainset_wav_files)
    num_validset_wav_files = len(validset_wav_files)

    create_dir(args.save_path)
    print("**********  Start processing and writing trainset  **********")
    trainset_records_filepath = os.path.join(args.save_path, args.trainset_filename)
    total_trainset_slices = 0
    begin_time = timeit.default_timer()
    trainset = []
    for file_count, trainset_wav_file in enumerate(trainset_wav_files):
        print("Processing trainset wav file {}/{} {}{}".format(file_count + 1, num_trainset_wav_files, 
                                                               trainset_wav_file, " " * 10), end="\r")
        sys.stdout.flush()
        num_slices, trainset = wav_processing(trainset_wav_file, window_size, trainset)
        total_trainset_slices += num_slices
    print("**************   Post-processing trainset   **************")
    need_to_cut = total_trainset_slices % global_batch_size
    if need_to_cut > 0:
        trainset = trainset[:-1 * need_to_cut]
    trainset = torch.cat(trainset, dim=0)
    torch.save(trainset, trainset_records_filepath)
    end_time = timeit.default_timer() - begin_time
    print(" ")
    print("*" * 50)
    print("Total processing and writing time: {} s, num:{}".format(end_time, total_trainset_slices - total_trainset_slices % global_batch_size))
    print(torch.max(trainset), torch.min(trainset))
    print(" ")
    print(" ")
    del trainset, trainset_wav_files
    print("**********  Start processing and writing validset  **********")
    validset_records_filepath = os.path.join(args.save_path, args.validset_filename)
    total_validset_slices = 0
    begin_time = timeit.default_timer()
    validset = []
    for file_count, validset_wav_file in enumerate(validset_wav_files):
        print("Processing validset wav file {}/{} {}{}".format(file_count + 1, num_validset_wav_files, 
                                                               validset_wav_file, " " * 10), end="\r")
        sys.stdout.flush()
        num_slices, validset = wav_processing(validset_wav_file, window_size, validset)
        total_validset_slices += num_slices
    print("**************   Post-processing validset   **************")
    need_to_cut = total_validset_slices % global_batch_size
    if need_to_cut > 0:
        validset = validset[:-1 * need_to_cut]
    validset = torch.cat(validset, dim=0)
    torch.save(validset, validset_records_filepath)
    end_time = timeit.default_timer() - begin_time
    print(" ")
    print("*" * 50)
    print("Total processing and writing time: {} s, num:{}".format(end_time, total_validset_slices - total_validset_slices % global_batch_size))
    print(torch.max(validset), torch.min(validset))
    print(" ")
    print(" ")
