import argparse
import os
import torch
import torchaudio
import scipy.signal
import scipy.io.wavfile

def resampler(original_sample_rate, waveform, target_sample_rate=8000):
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")
    
    # parameter of frame
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride")
    
    # parameter of semantic coding and channel coding
    parser.add_argument("--sem_enc_outdims", type=list, default=[32, 32, 32, 32, 32, 32, 32, 32],
                        help="output dimension of SE-ResNet in semantic encoder.")
    # [64, 32, 16, 4]
    parser.add_argument("--chan_enc_filters", type=list, default=[8],
                        help="filters of CNN in channel encoder.")
    parser.add_argument("--chan_dec_filters", type=list, default=[32],
                        help="filters of CNN in channel decoder.")
    parser.add_argument("--sem_dec_outdims", type=list, default=[32, 32, 32, 32, 32, 32, 32, 32],
                        help="output dimension of SE-ResNet in semantic decoder.")
    parser.add_argument("--snrs", type=list, default=[0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22],
                        help="snrs")
        
    # epoch and learning rate
    parser.add_argument("--num_epochs", type=int, default=400, help="training epochs.")
    
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
    # path of tfrecords files
    parser.add_argument("--trainset_path", type=str, default="./dataset/deepsc/trainset_t1.pth",
                        help="records path of trainset.")
    parser.add_argument("--validset_path", type=str, default="./dataset/deepsc/validset_t1.pth",
                        help="records path of validset.")
    parser.add_argument("--channel_type", type=str, default="awgn",
                        help="awgn or rayleigh or rician.")
    

    args = parser.parse_args()
    
    return args
