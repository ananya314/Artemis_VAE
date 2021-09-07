import pandas as pd
#!pip install wordsegment
from wordsegment import load, segment

path = "/content/drive/MyDrive/Lumiere/Dataset/artemis_data.csv"

# artemis_data.csv is a csv file that contains all captions from the ArtEmis dataset. It has two columns TEXT and LABEL that both contain ArtEmis captions. 
# TEXT and LABEL are the same because a VAE learns to reconstruct data. Hence, the expected output (label) is same as the input.

artemis_data = pd.read_csv(path)
load()

# Few captions in the ArtEmis dataset do not have spaces between words and look like: anexamplesentence. 
# wordsegment helps to add spaces in this circumstance.

while i < artemis_data.shape[0]:
  if i % 100 == 0:
    print(i)
  artemis_data["TEXT"][i] = " ".join(segment(artemis_data["TEXT"][i])) + "."
  artemis_data["LABEL"][i] = " ".join(segment(artemis_data["TEXT"][i])) + "."

  i = i + 1

# saving the preprocessed dataset as actual_sentences.csv
import os
path = "/content/drive/MyDrive/Lumiere/"
os.chdir(path)
artemis_data.to_csv("actual_sentences.csv")


