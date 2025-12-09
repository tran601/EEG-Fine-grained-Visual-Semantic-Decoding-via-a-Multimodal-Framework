import numpy as np
import os
import random
from tqdm.auto import tqdm
from loss import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import (
    Model,
    Model_NoTemporal,
    Model_NoSpatial,
    Model_NoST,
    Model_OnlyTemporal,
    Model_OnlySpatial,
)
from dataset import prepare_dataloaders
from torch.autograd import Variable


cudnn.benchmark = True

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print(device, "——", GPU_device.name, "——training")

FloatTensor = torch.cuda.FloatTensor
batch_size = 64
train_dataloader, test_dataloader = prepare_dataloaders(batch_size=batch_size, type=40)

epochs = 100
model = Model()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)


alpha = 0.3

for epoch in range(1, epochs + 1):
    losses = {"train": 0, "test": 0}
    total_num = {"train": 0, "test": 0}
    model.train()
    torch.set_grad_enabled(True)
    for i, (eeg, _, text_embeds, _) in enumerate(tqdm(train_dataloader)):
        text_embeds = text_embeds.float()
        eeg = Variable(eeg.type(FloatTensor)).to(device)
        eeg_embeds = model(eeg)
        total_num["train"] += eeg.shape[0]

        loss = (1-alpha) * mse_loss(eeg_embeds, text_embeds) + alpha * cosine_similarity_loss(
            eeg_embeds, text_embeds
        )
        # loss = mse_loss(eeg_embeds, text_embeds)
        # loss = cosine_similarity_loss(eeg_embeds, text_embeds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses["train"] += loss.item()

    model.eval()
    torch.set_grad_enabled(False)
    for i, (eeg, _, text_embeds, _) in enumerate(test_dataloader):
        text_embeds = text_embeds.float()
        eeg = Variable(eeg.type(FloatTensor)).to(device)
        eeg_embeds = model(eeg)
        total_num["test"] += eeg.shape[0]

        loss = (1-alpha) * mse_loss(eeg_embeds, text_embeds) + alpha * cosine_similarity_loss(
            eeg_embeds, text_embeds
        )
        # loss = mse_loss(eeg_embeds, text_embeds)
        # loss = cosine_similarity_loss(eeg_embeds, text_embeds)
        losses["test"] += loss.item()

    print(
        f"epoch {epoch}   train loss: {losses['train'] / total_num['train']:.5f}   test loss: {losses['test'] / total_num['test']:.5f}"
    )
    if epoch % epochs == 0:
        torch.save(model, f"save_model/epoch_{epoch}.pth")
        print("=============model saved=============")
