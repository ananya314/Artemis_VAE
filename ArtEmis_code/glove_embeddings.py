import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

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

path = "/content/drive/MyDrive/Lumiere/Dataset/actual_sentences.csv"
artemis_data = pd.read_csv(path)

TEXT = torchtext.legacy.data.Field(sequential=True, use_vocab=True,
                                   tokenize='spacy',  # default splits on whitespace
                                   tokenizer_language='en', init_token="<sos>", eos_token="<eos>"  # not used lower=True
                                   )

LABEL = torchtext.legacy.data.Field(sequential=True, use_vocab=True,
                                    tokenize='spacy',  # default splits on whitespace
                                    tokenizer_language='en', init_token="<sos>", eos_token="<eos>"
                                    )

fields = [("TEXT", TEXT), ("LABEL", LABEL)]

dataset = torchtext.legacy.data.TabularDataset(
    path=path, format='csv',
    skip_header=True, fields=fields)

train_data, test_data = dataset.split(
    split_ratio=[0.99, 0.01],  # 99/1 split
    random_state=random.seed(random_seed))  # change this

TEXT.build_vocab(train_data, min_freq=2,
                 vectors="glove.6B.100d")  # vectors="glove.6B.100d" not working <urlopen error [Errno 111] Connection refused>
LABEL.build_vocab(train_data, min_freq=2, vectors="glove.6B.100d")  # same here

train_loader, test_loader = torchtext.legacy.data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=False,
    sort_key=lambda x: len(x.TEXT),
    device=DEVICE
)
