# Investigating the Diversity of Response to Artwork through Latent Space Analysis of Captions from the ArtEmis Dataset
This project examines the latent space encodings of an LSTM beta-VAE trained on the [ArtEmis dataset](https://www.artemisdataset.org/) to generate emotion-driven text. The ArtEmis dataset is a collection of captions of artwork based on the WikiArt dataset. It contains subjective captions for each artwork and focuses on the personal interpretations and descriptions of emotions elicited by a particular work of art. 

After training the VAE model to generate emotion-driven text using the ArtEmis dataset, we draw conclusions about specific artworks and genres in the dataset by examining the similarity between their latent encodings. We quantitatively evaluate the diversity of response elicited by a work of art by analyzing the learned latent space representation of the VAE. After organizing the dataset by artwork, analysis is done by evaluating the Euclidean distance between the latent space representation of captions in the dataset and their respective centroid. We sort these distances and use them as a metric to evaluate the diversity of responses elicited by a particular work of art. 

In addition, we qualitatively examine the latent representations of captions in the dataset using PCA. We found that the top five styles of art that elicited the most diverse responses were analytical cubism, synthetic cubism, pointillism, contemporary realism, and action painting. 

## An Example from the ArtEmis Dataset 
<p align="center">
  <img src="https://github.com/ananya314/Artemis_VAE/blob/main/Images/art.png" alt="ArtEmis Dataset Example" width="300"/>
</p>

## Experimental Design
<p align="center">
  <img src="https://github.com/ananya314/Artemis_VAE/blob/main/Images/exp_design.png" alt = "Flowchart" width="600"/>
</p>

