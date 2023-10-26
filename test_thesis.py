# -*- coding: utf-8 -*-
"""
Created on 2023/10/12
Author: Hailong Lin
File: train.py
Email: linhl@emnets.org
Last modified: 2023/10/12 19:35
"""
import os
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from metrics import pesq_score, sdr_score, stoi_score
from model_thesis import DeepSCThesis
from utils import create_dir, parse_args
from log import create_logger
from custom_dataset import CustomDataset
from config import Config
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
opt = parse_args()
name = "thesis_with_noise"
logger = create_logger(filename=f"./log/{name}_test.log", ltype="a")
print("Called with args:", opt)
config = Config
channel_type = opt.channel_type
config.channel["type"] = channel_type
print("config:", config.channel)

deepsc = DeepSCThesis(opt, config=config).to(device)
validset = torch.load(opt.validset_path)
valid_dataset = CustomDataset(validset)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                              drop_last=True, num_workers=2)
logger.info(f"Dataset Load!!!, valid_num:{len(valid_dataloader)}")
mse_loss = torch.nn.MSELoss()


common_dir = "results/"
create_dir(common_dir)
saved_model = common_dir + f"saved_model_{name}/"
create_dir(saved_model)
logger.info("*****************   start test   *****************")
deepsc.transmitter.load_state_dict(torch.load(saved_model + "deepsc_transmitter_min.pth")['state_dict'])
deepsc.receiver.load_state_dict(torch.load(saved_model + "deepsc_receiver_min.pth")['state_dict'])
deepsc.eval()

valid_loss = 0.0
logger.info(f"channel_type: {channel_type}")
valid_loss_file = saved_model + f"{channel_type}.npz"
valid_loss_all = []

valid_loss = 0.0
# snrs = [x for x in range(0, 32, 2)]
snrs = opt.snrs
PESQs = []
SDRs = []
STOIs = []
CHANNEL_USAGEs = []

# record the valid time for each epoch
start = time.time()
sdrs = []
stois = []
pesqs = []

for snr in snrs:
    PESQ = []
    SDR = []
    STOI = []
    channel_usages = []
    channel_us = 0
    for step, _input in enumerate(tqdm(valid_dataloader)):
        # train step
        # if step == 1:
        #     break
        with torch.no_grad():
            x = _input.to(device)
            _output, channel_usage = deepsc(x, snr=snr)
            channel_usage /= x.shape[0]
            channel_usages.append(channel_usage)
        # try:
            x = x.cpu().detach()
            _output = _output.cpu().detach()
            for i in range(x.shape[0]):
                xx = x[i].unsqueeze(0).contiguous()
                __output = _output[i].unsqueeze(0).contiguous()
                score = pesq_score(opt.sr, xx, __output)
                sdr = sdr_score(xx, __output)
                stoi = stoi_score(xx, __output, opt.sr)
                PESQ.append(score)
                SDR.append(sdr[0])
                STOI.append(stoi)
            # print(channel_usage, stoi, sdr, score)
        # except:
        #     pass
    mean_channel_usage = sum(channel_usages) / len(channel_usages)
    mean_pesq = sum(PESQ) / len(PESQ)
    mean_sdr = sum(SDR) / len(SDR)
    mean_stoi = sum(STOI) / len(STOI)
    CHANNEL_USAGEs.append(mean_channel_usage)
    PESQs.append(mean_pesq)
    SDRs.append(mean_sdr)
    STOIs.append(mean_stoi)
    logger.info(f"snr:{snr}, pesq_mean:{mean_pesq}, mean_sdr:{mean_sdr}, mean_stoi:{mean_stoi}")
    np.savez(valid_loss_file, pesq=np.array(PESQs),
                sdr=np.array(SDRs), stoi=np.array(STOIs), snr=np.array(snrs), channel_usage=np.array(CHANNEL_USAGEs))
    
