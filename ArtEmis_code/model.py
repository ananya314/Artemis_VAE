from glove_embeddings import TEXT, LABEL, train_loader, test_loader

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

torch.backends.cudnn.deterministic = True

random_seed = 42
torch.manual_seed(random_seed)
BATCH_SIZE = 32 # originally 128
NUM_EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
LAT_DIM = 100 # check again (amnt of data)
N_LAYERS = 1 # N layers = 2 is complicated
ENC_DROPOUT = 0.5 # 0.5 may be too much
DEC_DROPOUT = 0.5


# ENCODER
class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, latent_dim: int, n_layers: int, dropout: float):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)  # [25805-> 256] [len(TEXT.vocab)->embed dim]
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)  # [256 -> 512] [embed dim -> hidden dim]

        self.z_mean = torch.nn.Linear(hid_dim, latent_dim)  # [512 -> 100] [hidden dim -> latent dim]
        self.z_logvar = torch.nn.Linear(hid_dim, latent_dim)  # [512 -> 100] [hidden dim -> latent dim]

    def reparameterise(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()  # sample from normal distribution
        return eps.mul(std).add_(mu)

    def forward(self, x: torch.LongTensor):
        embedded = self.embedding(x)  # [sentence len, batch size, embed dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        mu = self.z_mean(hidden)
        logvar = self.z_logvar(hidden)
        z = self.reparameterise(mu, logvar)

        return z, mu, logvar, hidden, cell  # we do not need to return mu and logvar


# DECODER
class Decoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, latent_dim: int, n_layers: int, dropout: float):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.linear1 = torch.nn.Linear(latent_dim, hid_dim)  # [100-> 256] [latent dim -> embed dim]
        # embed?
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn_decoder = nn.LSTM(emb_dim, hid_dim, n_layers,
                                   dropout=dropout)  # [256 -> 512] [embed dim -> hidden dim] # not sure about this here
        self.linear2 = torch.nn.Linear(hid_dim, input_dim)  # [512 -> 25805] [hidden dim -> len(TEXT.vocab)]

    def forward(self, epoch_num, trg, hidden, cell): #NOTE: epoch_num does NOT refer to the epoch number. It refers to the value of "i" in the VAE.
        # if (epoch_num == 1):
        #   output_lat = self.linear1(trg.unsqueeze(0)) #requires 3 dimensions [1, batch size, emb dim] sentence len = 1 (decode one at a time)
        #    output_lstm, (hidden, cell_state) = self.rnn_decoder(output_lat, (hidden, cell))
        # else:
        output_lat = self.embedding(trg.unsqueeze(0))
        # print("output_lat: ", output_lat.shape)

        if (epoch_num == 1):
            hidden = self.linear1(hidden)
            # print("hidden z: ", hidden.shape)

        # print(epoch_num)

        output_lstm, (hidden, cell) = self.rnn_decoder(output_lat, (hidden, cell))
        # print("output_lstm: ", output_lstm.shape, "hidden lstm: ", hidden.shape)

        output = self.linear2(output_lstm.squeeze(0))  # squeezing out the [1,]
        return output, hidden, cell_state


class VAE(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):

        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers!'

    def forward(self, x: torch.Tensor):
        max_len, batch_size = x.shape
        v_size = self.decoder.input_dim  # size of vocab
        outputs = torch.zeros(max_len, batch_size, v_size).to(self.device)  # store outputs

        z, mu, logvar, hidden, cell = self.encoder(x)

        trg = x[0]

        # decoding one at a time
        for i in range(1, max_len):


            if i == 1:
                hidden = z
                #print("hidden: ", hidden.shape) # debugging
                #cell = torch.zeros((1, BATCH_SIZE, 512)).to(DEVICE)  # try initial cell state = 0

            prediction, hidden, cell = self.decoder(i, trg, hidden, cell)
            outputs[i] = prediction

            trg = prediction.argmax(1)  # no teacher forcing

        return outputs, mu, logvar

