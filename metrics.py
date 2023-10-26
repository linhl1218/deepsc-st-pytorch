# -*- coding: utf-8 -*-
"""
Created on 2023/09/27 14:17
@author: Hailong Lin
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from pesq import pesq
import librosa
import museval
import torchaudio
from pystoi import stoi
import mir_eval

def sdr_score(origin, fake):
    origin = origin.view(-1).numpy()
    fake = fake.view(-1).numpy()
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(origin, fake)
    return sdr

def stoi_score(origin, fake, sample_rate=8000):
    # 0.0~1.0
    origin = origin.view(-1).numpy()
    fake = fake.view(-1).numpy()
    score = stoi(origin, fake, sample_rate)
    return score

def pesq_score(sample_rate, origin, fake):
    # -0.5~4.5
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        resampler_origin = resampler(origin)
        resampler_fake = resampler(fake)
    else:
        resampler_origin = origin
        resampler_fake = fake
    origin_np = resampler_origin.view(-1).numpy()
    fake_np = resampler_fake.view(-1).numpy()
    # print(target_sample_rate, origin_np.shape, fake_np.shape, np.max(origin_np), np.min(origin_np),
    #       np.max(fake_np), np.min(fake_np))
    # print(resampler_origin.view(-1).numpy().shape, resampler_fake.view(-1).numpy().shape,
    #       origin_np.max(), fake_np.max(), origin_np.min(), fake_np.min())
    score = pesq(target_sample_rate, origin_np, fake_np)
    return score

# def sdr_score(origin, fake):
#     origin_np = origin.view(-1).numpy()
#     fake_np = fake.view(-1).numpy()
#     score = museval.metrics.bss_eval(origin_np, fake_np)
#     return score

def ssim_score(img1, img2):
    """_summary_
    Input two batchs or two images, return the SSIM score.
    """
    num_dimensions = img1.ndim
    shape = img1.shape
    if num_dimensions == 3:
        # img1 = img1.permute(shape[1], shape[2], shape[0])
        # img2 = img2.permute(shape[1], shape[2], shape[0])
        score = structural_similarity(img1, img2, multichannel=True)
    elif num_dimensions == 4:
        # img1 = img1.permute(shape[0], shape[2], shape[3], shape[1])
        # img2 = img2.permute(shape[0], shape[2], shape[3], shape[1])
        scores = np.zeros(shape[0])
        for i in range(shape[0]):
            scores[i] = structural_similarity(img1[i], img2[i], multichannel=True)
        score = np.mean(scores)
    else:
        score = structural_similarity(img1, img2)
    return score

def psnr_score(img1, img2, data_range=255):
    """_summary_
    Input two batchs or two images, return the PSNR score.
    """
    num_dimensions = img1.ndim
    shape = img1.shape
    if num_dimensions == 4:
        scores = np.zeros(shape[0])
        for i in range(shape[0]):
            scores[i] = peak_signal_noise_ratio(img1[i], img2[i], data_range=data_range)
        score = np.mean(scores)
    else:
        score = peak_signal_noise_ratio(img1, img2, data_range=data_range)
    return score

def mse_score(img1, img2):
    """_summary_
    Compute the mean-squared error between two images.
    """
    num_dimensions = img1.ndim
    shape = img1.shape
    if num_dimensions == 4:
        scores = np.zeros(shape[0])
        for i in range(shape[0]):
            scores[i] = mean_squared_error(img1[i], img2[i])
        score = np.mean(scores)
    else:
        score = mean_squared_error(img1, img2)
    return score

if __name__=="__main__":
    pass
