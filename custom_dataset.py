import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  
        self.num = data.shape[0]
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = self.data[idx, :]  # 获取一个样本，大小为 (1, 16800)
        return sample
