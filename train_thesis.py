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
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
from model_thesis import DeepSCThesis
from utils import create_dir, parse_args
from log import create_logger
from custom_dataset import CustomDataset
from config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parse_args()
channel_type = opt.channel_type
name = "thesis_with_noise"
logger = create_logger(filename=f"./log/{name}_{channel_type}_train.log", ltype="a")
print("Called with args:", opt)
config = Config
config.channel['type'] = channel_type
config.logger = logger
deepsc = DeepSCThesis(opt, config=config).to(device)
trainset = torch.load(opt.trainset_path)
validset = torch.load(opt.validset_path)
train_dataset = CustomDataset(trainset)
valid_dataset = CustomDataset(validset)
# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size * 2, shuffle=True,
                              drop_last=True, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                              drop_last=True, num_workers=2)
logger.info(f"Dataset Load!!!, train_num: {len(train_dataloader)}, valid_num:{len(valid_dataloader)}")
mse_loss = torch.nn.MSELoss()
LR_START = opt.lr
LR_FINE = 1e-4
MOMENTUM = 0.1

# optimizer = SGD(deepsc.parameters(), lr=LR_START, momentum=MOMENTUM)

optimizer = Adam(deepsc.parameters(), lr=LR_START)
scheduler = ReduceLROnPlateau(optimizer, 'min') 

common_dir = "results/"
create_dir(common_dir)
saved_model = common_dir + f"saved_model_{name}/"
create_dir(saved_model)
saved_model += f"{channel_type}/"
create_dir(saved_model)
CONTINUES = 0
# create files to save train loss
logger.info("*****************   start train   *****************")
MIN_VALID_LESS = 1e1
P=opt.P
LR_FINE_CHANGE = 1

# deepsc.transmitter.load_state_dict(torch.load(saved_model + "deepsc_transmitter_min.pth")['state_dict'])
# deepsc.receiver.load_state_dict(torch.load(saved_model + "deepsc_receiver_min.pth")['state_dict'])

for epoch in range(opt.num_epochs):
    deepsc.train()
    if (epoch + 1) % 280 == 0 and LR_FINE_CHANGE:
        LR_FINE_CHANGE = 0
        optimizer = SGD(deepsc.parameters(), lr=LR_FINE, momentum=MOMENTUM)
        scheduler = ReduceLROnPlateau(optimizer, 'min')        
        logger.info(f"Change optimizer Adam to SGD, lr from {LR_START} to {LR_FINE}")
    if epoch < CONTINUES:
        continue

    # train_loss for each epoch
    train_loss = 0.0
    # record the train time for each epoch
    start = time.time()
    for step, _input in enumerate(train_dataloader):
        # train step
        x = _input.to(device)
        _output, _ = deepsc(x)
        loss_value = mse_loss(x, _output)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        loss_float = float(loss_value)
        train_loss += loss_float
    train_loss /= (step + 1)
    # append one epoch loss value
    # print log
    log = "train epoch {}/{}, train_loss = {:.010f}, time = {:.010f}"
    message = log.format(epoch + 1, opt.num_epochs, train_loss, time.time() - start)
    logger.info(message)
    ##########################    valid    ##########################
    deepsc.eval()
    # valid_loss for each epoch
    valid_loss = 0.0
    # record the valid time for each epoch
    start = time.time()
    for step, _input in enumerate(valid_dataloader):
        # train step
        with torch.no_grad():
            x = _input.to(device)
            _output, _ = deepsc(x)
            loss_value = mse_loss(x, _output)
        loss_float = float(loss_value)
        if scheduler is not None:
            scheduler.step(loss_value)
        # Calculate the accumulated valid loss value
        valid_loss += loss_float
    # average valid loss for each epoch
    valid_loss /= (step + 1)
    # append one epoch loss value
    # print log
    log = "valid epoch {}/{}, valid_loss = {:.010f}, time = {:.010f}"
    message = log.format(epoch + 1, opt.num_epochs, valid_loss, time.time() - start)
    logger.info(message)
    ###################    save the train network    ###################
    if (epoch + 1) % 50 == 0:
        saved_model_path = saved_model + "deepsc_transmitter_{}_epochs.pth".format(epoch + 1)
        torch.save({'state_dict': deepsc.transmitter.state_dict(), }, saved_model_path)
        saved_model_path = saved_model + "deepsc_receiver_{}_epochs.pth".format(epoch + 1)
        torch.save({'state_dict': deepsc.receiver.state_dict(), }, saved_model_path)
        logger.info(f"Per 50, model saved, epoch:{epoch + 1}")
    if valid_loss < MIN_VALID_LESS:
        saved_model_path = saved_model + "deepsc_transmitter_min.pth"
        torch.save({'state_dict': deepsc.transmitter.state_dict(), }, saved_model_path)
        saved_model_path = saved_model + "deepsc_receiver_min.pth"
        torch.save({'state_dict': deepsc.receiver.state_dict(), }, saved_model_path)
        logger.info(f"Comparsion, model saved, epoch:{epoch + 1}")
        MIN_VALID_LESS = valid_loss
    if (epoch + 1) % 10 == 0:
        saved_model_path = saved_model + "deepsc_transmitter_latest.pth"
        torch.save({'state_dict': deepsc.transmitter.state_dict(), }, saved_model_path)
        saved_model_path = saved_model + "deepsc_receiver_latest.pth"
        torch.save({'state_dict': deepsc.receiver.state_dict(), }, saved_model_path)
        logger.info(f"Latest, model saved, epoch:{epoch + 1}")
        
