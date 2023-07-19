# Hateful-Memes-Detection
This repository contains all code that has been used for my thesis project. The project aims to explore the impact of text encoder selection, feature extraction by image captioning, hyperparameter optimization and ensemble learning on hateful meme detection, and build a multimodal hateful memes classification system based on CLIP model.

# Getting started: Preparation before reproducing all codes

## Download Dataset
The dataset used for my thesis project is the Facebook Hateful Memes Dataset, which is the largest annotated memes dataset that contains more than 10K memes with gold labels. Since the dataset contains some offensive and distasteful memes that may be disturbing to some people, Facebook AI Research strictly restrict the dissemination of this dataset. Anyone who wants to use this dataset for academic research must fill out a form and agree to the dataset license agreement on this website: https://hatefulmemeschallenge.com/#download

## Environment Setting
All code experiments were performed using Nvidia A100/V100 GPU provided by Google Colaboratory Pro+ (Colab Pro+). If you want to reproduce my code in Google Colaboratory without changing any file paths, please upload the dataset you downloaded (hateful_memes.zip) along with my code to your Google Drive root directory and then run my code step by step in Google Colaboratory. If you want to run my code locally, please change all of the file paths in my code to your local file paths and install all required packages from the requirements.txt file. (Make sure to work with Python 3.10.12)

## Evaluation Metric and Current Benchmark
The evaluation metric used to report the performance of the multimodal classification model is the Area Under the Receiver Operating Characteristics curve (AUROC) on validation set (dev_seen) and test set (test_seen). According to the [Hateful Memes Challenge Report](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/) provided by Meta AI, the performance of trained annotators on the test set is 82.65. We recorded the outstanding performance of various SOTA models published from various top conferences in NLP/Multimedia in the last three years. The detailed statistics are shown in the table below:

|    Model     |  Validation AUROC  |  Test AUROC  |     Publication     |
| ------------ | ------------------ | ------------ | ------------------- |
| DisMultiHate |        82.8 (dev_seen)       |  not given   | [ACM Multimedia 2021](https://dl.acm.org/doi/10.1145/3474085.3475625) |
| Hate-CLIPper |        81.55 (dev_seen)      |    85.8 (test_seen)     | [EMNLP 2022 NLP4PI Workshop](https://aclanthology.org/2022.nlp4pi-1.20/) |
|  PromptHate  |        81.45 (dev_seen)      |  not given   | [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.22/) |
|   MemeFier   |        80.1 (dev_seen)       |  not given   | [ACM ICMR 2023](https://dl.acm.org/doi/abs/10.1145/3591106.3592254) |
|     CDKT     |        79.89 (dev_seen)      |    83.74 (test_seen)    | [ACM Multimedia 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548255) |
|     CES      |        78.29 (dev_seen)      |    78.9 (test_seen)     | [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.738/) |

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

| Model Version  | Validation AUROC |  Model Size  |  Colab Links  |
| -------------- | ---------------- | ------------ | ------------- |
|    ViT-L/14    |      81.59       |    890MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nbQwAKPVymAGFfOJL-8DtlfUl5V2R5KB) |
| ViT-L/14@336px |      80.45       |    891MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I_4F593MJD_YVDxqwnbPKqU2h_ntLW6g) |
|    ViT-B/16    |      74.27       |    335MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MzbgFUcyMI_zrtFwCWVTfbVXFOV3cHxM) |
|    ViT-B/32    |      73.44       |    338MB     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1js683AnC-r0dlxn7khlDZV7C6rclCOhN) |

After comparison, we find that the CLIP model of ViT-L/14 version achieves the highest validation AUROC (81.59), so we will perform further experiments based on this version of model. We also tested the baseline model with test_seen set and achieved an AUROC of **81.98**  

## 3. Replacing text encoder to RoBERTa  
The folder contains the complete experiments for replacing the underlying text encoder of CLIP models to RoBERTa. Specifically, we tried two different versions of the pretrained RoBERTa model. The first is the RoBERTa-Large model from the HuggingFace Transformers library. The second is a domain-specific RoBERTa-base model trained on ~58M tweets and finetuned for offensive language identification with the [TweetEval benchmark](https://aclanthology.org/2020.findings-emnlp.148/), which consists in identifying whether some form of offensive language is present in a tweet. Since the text features of memes are similar to tweets in that they are short texts and contain colloquial expressions, we expect this domain-specific language model to help us better encode the text in memes.

| Text Encoder | Validation AUROC | Test AUROC |  Colab Links  |
| ------------ | ---------------- | ---------- | ------------- |
| [RoBERTa-Large](https://huggingface.co/roberta-large) | 76.79 | 80.92 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vd3M0wQct6gG8qhQ7O5uSa7f3WaeWBGV) |
| [Twitter-RoBERTa-base-offensive](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive) | 78.47 | 81.33 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vi6nKfbU5f5cNj_AZhQrml5JiM71U6BE) |
| CLIP baseline | 81.59 | 81.98 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nbQwAKPVymAGFfOJL-8DtlfUl5V2R5KB) |

It is not difficult to see that replacing the underlying text encoder of the CLIP model with RoBERTa-Large will lead to a decrease in the classification performance of the CLIP model. Although the RoBERTa-base model trained and fine-tuned on specific domains outperforms RoBERTa-Large model, it is still not enough to compete with the original text encoder of CLIP.

## 4. Feature Extraction by Image Captioning
The folder contains the complete experiments for feature extraction by image captioning. Firstly, we perform data pre-processing to detect and remove texts from meme images and save the clean images for feature extraction. Then we apply pre-trained image captioning model [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) to generate textual description of clean images. The generated captions will be saved as corresponding CSV file for integrating image captioning features as additional textual inputs of the CLIP model for hateful memes classification. However, the results show that the generated captioning features do not help the CLIP model to improve its performance:

| Image Captioning Model | Generated Captions | Validation AUROC | Test AUROC |  Colab Links  |
| ---------------------- | ------------------ | ---------------- | ---------- | ------------- |
| [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) | [BLIP_caption.csv]() | 81.03 | 81.71 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ayYbF5O40KTzbxYJzTpwuSEllLZZEToB) |

After careful inspection of the generated captions, we find that the quality of captions generated by BLIP model is not very high. For example, the model generated an exception word "arafed" at the beginning of some captions, which appears 2,734 times in total. These erroneous captions introduce bias and noise in the classification model and affect its classification performance.  

We then tried to apply another pre-trained image captioning model [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b-coco) for captions generation, which is an upgraded version of the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) model. It leverages a decoder-based large language model [OPT](https://huggingface.co/facebook/opt-2.7b) with 2.7 billion parameters to help generate higher quality captions. The results show that the generated captioning features can help the CLIP model to improve its performance:

| Image Captioning Model | Generated Captions | Validation AUROC | Test AUROC |  Colab Links  |
| ---------------------- | ------------------ | ---------------- | ---------- | ------------- |
| [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b-coco) | [BLIP_2_caption.csv]() | 81.79 | 82.74 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17-GtNCnQoXtCIvyfexCj_fOlokKYRlOy) |

## 5. Hyperparameter optimization + Ensemble Learning
The folder contains the complete experiments for hyper-parameters tuning over the dropout rate, learning rate, activation function, optimizer and scheduler type and save all best performing models based on the validation AUROC score. We then perform soft voting method for ensemble learning by averaging the predictions of best performing models. The AUROC score showed that the CLIP model can be improved by applying this strategy.

| Hyperparameter Optimization | Validation AUROC | Test AUROC |  Colab Links  |
| ------------ | ---------------- | ---------- | ------------- |
| Ensemble Learning | 83.46 | 83.23 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WaNIYdm4jZEREKp3u_XWZyRKYDX-XAux) |
