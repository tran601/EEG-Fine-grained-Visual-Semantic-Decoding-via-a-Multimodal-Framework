from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from dataset import EEGDataset_40, EEGDataset_4
import torch
from tqdm.auto import tqdm


cudnn.benchmark = True
seed = 2023
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
FloatTensor = torch.cuda.FloatTensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print(device, "——", GPU_device.name, "——feature extraction")

dataset = EEGDataset_40(split="train")
dataloader = DataLoader(dataset, batch_size=64, drop_last=False, shuffle=False)
model_path = "save_model/epoch_50.pth"
model = torch.load(model_path)
model.eval()
model.to(device)


feature_data = []

output_path = (
    "../data/ImageNet_40/uncondition_caption/embeddings/eeg_embedding_show.npy"
)
for i, (eeg, text, _, img_label) in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        eeg = Variable(eeg.type(FloatTensor)).to(device)
        x = model(eeg)  # [b, 77, 768]

    for j in range(len(eeg)):
        one_data = {
            "eeg": x[j].cpu().data.numpy(),
            "text": text[j],
            "img_label": img_label[j].cpu().data.numpy(),
        }
        feature_data.append(one_data)

print(len(feature_data))
np.save(output_path, np.array(feature_data), allow_pickle=True, fix_imports=True)

print("=============已完成=============")
