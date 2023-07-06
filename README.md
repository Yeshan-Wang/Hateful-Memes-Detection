# Hateful-Memes-Detection
This repository contains all code that has been used for my thesis project. The project aims to explore the impact of data augmentation, feature extraction by image captioning, text encoder selection, and ensemble learning on hateful meme detection, and build a State-of-the-Art multimodal hateful memes classification system based on CLIP model.

# Getting started: Preparation before reproducing all codes

## Download Dataset
The dataset used for my thesis project is the Facebook Hateful Memes Dataset, which is the largest annotated memes dataset that contains more than 10K memes with gold labels. Since the dataset contains some offensive and distasteful memes that may be disturbing to some people, Facebook AI Research strictly restrict the dissemination of this dataset. Anyone who wants to use this dataset for academic research must fill out a form and agree to the dataset license agreement on this website: https://hatefulmemeschallenge.com/#download

## Environment Setting
All code experiments were performed using Nvidia A100 GPU provided by Google Colaboratory Pro+ (Colab Pro+). If you want to reproduce my code in Google Colaboratory without changing any file paths, please upload the dataset you downloaded (hateful_memes.zip) along with my code to your Google Drive root directory and then run my code step by step in Google Colaboratory. If you want to run my code locally, please change all of the file paths in my code to your local file paths and install all required packages from the requirements.txt file. (Make sure to work with Python 3.10.12)

## Evaluation Metric and Current Benchmark
The evaluation metric used to report the performance of the multimodal classification model is the Area Under the Receiver Operating Characteristics curve (AUROC) on validation set (dev_seen) and test set (test_seen). According to the [Hateful Memes Challenge Report](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/) provided by Meta AI, the performance of trained annotators on the test set is 82.65. We recorded the outstanding performance of various SOTA models published from various top conferences in NLP/Multimedia in the last three years. The detailed statistics are shown in the table below:

|    Model     |  Validation AUROC  |  Test AUROC  |     Publication     |
| ------------ | ------------------ | ------------ | ------------------- |
| DisMultiHate |        82.8        |  not given   | [ACM Multimedia 2021](https://dl.acm.org/doi/10.1145/3474085.3475625) |
| Hate-CLIPper |        81.55       |    85.8      | [EMNLP 2022 NLP4PI Workshop](https://aclanthology.org/2022.nlp4pi-1.20/) |
|  PromptHate  |        81.45       |  not given   | [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.22/) |
|   MemeFier   |        80.1        |  not given   | [ACM ICMR 2023](https://dl.acm.org/doi/abs/10.1145/3591106.3592254) |
|     CDKT     |        79.89       |    83.74     | [ACM Multimedia 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548255) |
|     CES      |        78.29       |    78.9      | [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.738/) |

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
The folder contains the following four notebooks corresponding to the complete experiments for four different versions of CLIP models. We can compare the four models in the following table:  
 
| Model Version  | Validation AUROC | Test AUROC |  Model Size  |  Colab Links  |
| -------------- | ---------------- | ---------- | ------------ | ------------- |
| ViT-L/14@336px |      81.13       |   83.81    |    891MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NkA8TdIsofMHFJIXI-n6Ab3p1lVwr1It) |
|    ViT-L/14    |      81.62       |   82.41    |    890MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IEwz53Dn4qmE3R3WngCtYyIKzOPX77JG) |
|    ViT-B/16    |      74.29       |   79.94    |    335MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MzbgFUcyMI_zrtFwCWVTfbVXFOV3cHxM) |
|    ViT-B/32    |      73.04       |   76.89    |    338MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1js683AnC-r0dlxn7khlDZV7C6rclCOhN) |

After comparison, we find that the CLIP model of ViT-L/14@336px version achieves the highest test AUROC (83.81), so we will perform further experiments based on this version of model.  

## 3. Replacing text encoder to RoBERTa  
The folder contains the complete experiments for replacing the underlying text encoder of CLIP models to RoBERTa. Specifically, we tried two different versions of the pretrained RoBERTa model. The first is the RoBERTa-Large model from the HuggingFace Transformers library. The second is a RoBERTa-base model trained on ~58M tweets and finetuned for offensive language identification with the TweetEval benchmark.  

| Text Encoder | Validation AUROC | Test AUROC |  Colab Links  |
| ------------ | ---------------- | ---------- | ------------- |
| [RoBERTa-Large](https://huggingface.co/roberta-large) | 76.62 | 80.92 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vd3M0wQct6gG8qhQ7O5uSa7f3WaeWBGV) |
| [Twitter-RoBERTa-base-offensive](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive) | 78.02 | 82.11 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vi6nKfbU5f5cNj_AZhQrml5JiM71U6BE) |
| CLIP | 81.13 | 83.81 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NkA8TdIsofMHFJIXI-n6Ab3p1lVwr1It) |

It is not difficult to see that replacing the underlying text encoder of the CLIP model with RoBERTa-Large will lead to a decrease in the classification performance of the CLIP model. Although the RoBERTa-base model trained and fine-tuned on specific domains outperforms RoBERTa-Large model, it is still not enough to compete with the original text encoder of CLIP.

## 4. Text Augmentation.ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PN4PpQiRz8gcJ8Ah42yNfPJ-25i23Hb6)
We randomly select 50% of the memes from the original training set and use the data augmentation tool [AugLy](https://github.com/facebookresearch/AugLy) to replace some characters of original meme texts with random character noise that do not alter semantic. The newly created meme combinations are added to the original training set. Here's an example of original text vs augmented text for a given meme:  

| img | label | text |
| --- | ----- | ---- |
| img/18362.png	 | 0 | if they don't like it here they can leave! |
| img/18362.png	 | 0 | Ίf Ŧhey don't lίke Īt her£ τhey caŉ leavĖ! |

The test score of AUROC showed that the CLIP model can be improved by applying text augmentation strategy.

| Model | Validation AUROC | Test AUROC |
| ------------ | ---------------- | ---------- |
| CLIP+Text Augmentation | 81.03 | 84.58 |
| CLIP | 81.13 | 83.81 |

