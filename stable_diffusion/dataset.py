import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class EEG_EmbeddingDataset(Dataset):
    def __init__(self, name="eeg_embedding.npy"):
        path = (
            ""
            + name
        )
        self.data = np.load(path, allow_pickle=True)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"]
        text = self.data[i]["text"]
        img_label = self.data[i]["img_label"].item()

        return eeg, img_label, text


def prepare_dataloader(batch_size, name, ratio=1.0):
    dataset = EEG_EmbeddingDataset(name=name)

    total_size = len(dataset)
    sample_size = int(total_size * ratio)
    indices = list(range(sample_size))
    dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    return dataloader
