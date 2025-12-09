import numpy as np
import torch
from PIL import Image
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


# 从EEG提取特征对齐到文本嵌入
class EEGDataset_40(Dataset):
    def __init__(self, split="train"):
        path = (
            ""
            + split
            + ".npy"
        )
        self.data = np.load(path, allow_pickle=True)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460, :]
        text = self.data[i]["text"]
        text_embedding = self.data[i]["text_embedding"].squeeze(0)
        img_label = self.data[i]["img_label"]
        # 滤波
        HZ = 1000
        low_f, high_f = 1, 70
        b, a = signal.butter(2, [low_f * 2 / HZ, high_f * 2 / HZ], "bandpass")
        eeg = signal.lfilter(b, a, eeg).copy()

        return eeg, text, text_embedding, img_label


def prepare_dataloaders(batch_size, type=40):
    if type == 40:
        train_dataset = EEGDataset_40(split="train")
        test_dataset = EEGDataset_40(split="test")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )
    return train_dataloader, test_dataloader
