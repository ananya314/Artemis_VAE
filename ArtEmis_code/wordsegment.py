import pandas as pd
#!pip install wordsegment
from wordsegment import load, segment

path = "/content/drive/MyDrive/Lumiere/Dataset/artemis_data.csv"
artemis_data = pd.read_csv(path)
load()

while i < artemis_data.shape[0]:
  if i % 100 == 0:
    print(i)
  artemis_data["TEXT"][i] = " ".join(segment(artemis_data["TEXT"][i])) + "."
  artemis_data["LABEL"][i] = " ".join(segment(artemis_data["TEXT"][i])) + "."

  i = i + 1

import os
path = "/content/drive/MyDrive/Lumiere/"
os.chdir(path)
artemis_data.to_csv("actual_sentences.csv")


