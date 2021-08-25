from model import VAE, Encoder, Decoder, TEXT, LABEL, \
    train_loader, test_loader
import os
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
import os
import math
import time
import spacy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import List

from torchtext.legacy.data import Field, BucketIterator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42
torch.manual_seed(random_seed)
BATCH_SIZE = 32  # originally 128
NUM_EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
LAT_DIM = 100  # check again (amnt of data)
N_LAYERS = 1  # N layers = 2 is complicated
ENC_DROPOUT = 0.5  # 0.5 may be too much
DEC_DROPOUT = 0.5

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, LAT_DIM, N_LAYERS, ENC_DROPOUT).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, LAT_DIM, N_LAYERS, DEC_DROPOUT).to(DEVICE)
vae = VAE(encoder, decoder, DEVICE).to(DEVICE)

lr = 1e-3
optimizer = optim.Adam(vae.parameters(), lr=lr)
PAD_IDX = TEXT.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def rec_kl_loss(yhat, y, mu, logvar, beta):
    RL = criterion(yhat, y)
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    loss = RL + (beta * KLD)
    return loss


# Training and testing the VAE

epochs = 100
beta = 2
for epoch in range(0, epochs + 1):
    vae.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        outputs, mu, logvar = vae(batch.TEXT)
        outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
        label_flatten = batch.TEXT[1:].view(-1)

        loss = rec_kl_loss(outputs_flatten, label_flatten, mu, logvar, beta)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    os.chdir("/content/drive/MyDrive/Lumiere/Models/2/")
    torch.save(vae.state_dict(), f'model_{epoch + 1}.pt')  # os.chdir to drive

    print(f'============= Epoch: {epoch + 1} Average loss: {train_loss / len(train_loader.dataset):.4f}')
