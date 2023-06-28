# Hateful-Memes-Detection
This repository contains all code that has been used for my thesis project. The project aims to explore the impact of data augmentation, feature extraction by image captioning, text encoder selection, and ensemble learning on hateful meme detection, and build a State-of-the-Art multimodal hateful memes classification system based on CLIP model.

# Getting started: Preparation before reproducing all codes

## Download Dataset
The dataset used for my thesis project is the Facebook Hateful Memes Dataset, which is the largest annotated memes dataset that contains more than 10K memes with gold labels. Since the dataset contains some offensive and distasteful memes that may be disturbing to some people, Facebook AI Research strictly restrict the dissemination of this dataset. Anyone who wants to use this dataset for academic research must fill out a form and agree to the dataset license agreement on this website: https://hatefulmemeschallenge.com/#download

## Environment Setting
All code experiments were performed using Nvidia A100 GPU provided by Google Colaboratory Pro+ (Colab Pro+). If you want to reproduce my code in Google Colaboratory without changing any file paths, please upload the dataset you downloaded (hateful_memes.zip) along with my code to your Google Drive root directory and then run my code step by step in Google Colaboratory. If you want to run my code locally, please change all of the file paths in my code to your local file paths and install all required packages from the requirements.txt file. (Make sure to work with Python 3.10.12)

## Evaluation metric and Current Benchmark
The evaluation metric used to report the performance of the multimodal classification model is the Area Under the Receiver Operating Characteristics curve (AUROC) on validation set (dev_seen) and test set (test_seen). According to the report provided by Meta AI, the performance of trained annotators on the test set is 82.65. We recorded the outstanding performance of various SOTA models on this dataset in the last three years at various top conferences in NLP/Multimedia. The detailed statistics are shown in the table below:

|    Model     |  Validation AUROC  |  Test AUROC  |     Publication     |
| ------------ | ------------------ | ------------ | ------------------- |
| DisMultiHate |        82.8        |  Not given   | [ACM Multimedia 2021](https://dl.acm.org/doi/10.1145/3474085.3475625) |
| Hate-CLIPper |        81.55       |    85.8      | [EMNLP 2022 Workshop](https://aclanthology.org/2022.nlp4pi-1.20/) |
|  PromptHate  |        81.45       |  Not given   | [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.22/) |
|   MemeFier   |        80.1        |  Not given   | [ACM ICMR 2023](https://dl.acm.org/doi/abs/10.1145/3591106.3592254) |
|     CDKT     |        79.89       |    83.74     | [ACM Multimedia 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548255) |
|     CES      |        78.29       |    78.9      | [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.738/) |
|    TRICAN    |        78.27       |    79.81     | [IEEE IJCNN 2022](https://ieeexplore.ieee.org/abstract/document/9892164) |

# Reproducing all codes
**All codes/notebooks should be run in the following order:**

## 1. Exploratory Data Analysis.ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17zDK84NRg_9ZNYcrgdYeZE6CibihY-Bq)
The notebook performs exploratory data analysis on the Facebook Hateful Memes Dataset (especially the training set) in the following parts:
- The disrtibution of gold label (hateful/non-hateful) for train/dev/test set
- Exploring the internal structure of the dataset and some hateful/non-hateful samples in training set
- The disrtibution of number of characters in each meme from training set
- Exploring the most mentioned words of hateful/non-hateful memes in training set
- Display some images of hateful/non-hateful memes include the most mentioned words
- Exploring the most mentioned entities of hateful/non-hateful memes in training set

## 2. CLIP Model  
The folder contains the following four notebooks corresponding to the complete experiments for four different CLIP models:  
- CLIP(ViT-B/32).ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1js683AnC-r0dlxn7khlDZV7C6rclCOhN)
- CLIP(ViT-B/16).ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MzbgFUcyMI_zrtFwCWVTfbVXFOV3cHxM)
- CLIP(ViT-L/14).ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IEwz53Dn4qmE3R3WngCtYyIKzOPX77JG)
- CLIP(ViT-L/14@336px).ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NkA8TdIsofMHFJIXI-n6Ab3p1lVwr1It)  

We can compare the four models in the following table:  
|   Model Name   |  AUROC  |  Model Size  |  Time Cost of Encoding  |
| -------------- | ------- | ------------ | ----------------------- |
| ViT-L/14@336px |  83.81  |    891MB     |          3:10           |
|    ViT-L/14    |  82.41  |    890MB     |          2:46           |
|    ViT-B/16    |  79.94  |    335MB     |          2:32           |
|    ViT-B/32    |  76.89  |    338MB     |          2:28           |
