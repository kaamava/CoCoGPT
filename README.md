# CoCoGPT

## Introduction

Today, with the increasing risk of COVID-19 infection in the society, people who are suspected of COVID-19's symptoms or have a greater risk of infection for various reasons will have some anxiety about medical consultation. Relatively, the pressure on the medical system will rapidly increase in the short term. Opening up an online medical consultation platform can to some extent alleviate the current social pressure and public anxiety. However, at the same time, the professionalism of online consultation needs to be guaranteed. Therefore, the purpose of this study is to establish a medical dialogue model CoCoGPT (COVID-19 Consultant GPT) that can provide COVID-19 related consultations using professional medical consultation materials. The report utilizes and improves a Chinese medical dialogue dataset COVID-Dialogue-Chinese, which includes dialogue records between doctors and patients regarding COVID-19. In this study, the dataset were used to train a BERT-GPT based model.

## Dataset
COVID-Dialogue-Chinese is a Chinese medical dialogue dataset about COVID-19 and other types of pneumonia. Patients who are concerned that they may be infected by COVID-19 or other pneumonia consult doctors and doctors provide advice. There are 1088 consultations. Each consultation consists of

- ID
- URL
- Description of patient’s medical condition
- Dialogue
- (Optional) Diagnosis and suggestions.

The dataset is built from [Haodf.com](https://www.haodf.com/) and all copyrights of the data belong to [Haodf.com](https://www.haodf.com/).

## Model

### 1.Transformer

Transformer is an encoder-decoder architecture for sequence-to-sequence (seq2seq) modeling. Transformer is composed of a stack of building blocks, each consisting of a self-attention layer and a position-wise feed-forward layer. Residual connection is applied around each of the two sub-layers, followed by layer normalization.

### 2.GPT
The GPT model is a language model (LM) based on Transformer. Different from Transformer which defines a conditional probability on an output sequence given an input sequence, GPT defines a marginal probability on a single sequence. GPT-2 is an extension of GPT, which modifies GPT by moving layer normalization to the input of each sub-block and adding an additional layer normalization after the final self-attention block.

DialoGPT (Zhang et al., 2019) is a GPT-2 model pretrained on English Reddit dialogues. The dataset is extracted from comment chains in Reddit from 2005 till 2017, comprising 147,116,725 dialogue instances with 1.8 billion tokens.

### 3.BERT-GPT

BERT-GPT (Wu et al., 2019) is a model used for dialogue generation where pretrained BERT is used to encode the conversation history and GPT is used to generate the responses. While GPT focuses on learning a Transformer decoder for text generation purposes, BERT aims to learn a Transformer encoder for representing texts.

In BERT-GPT, the pretraining of the BERT encoder and the GPT decoder is conducted separately, which may lead to inferior performance.

BERT-GPT-Chinese is a BERT-GPT model pretrained on Chinese corpus. For the BERT encoder in BERT-GPT-Chinese, it is set to the Chinese BERT, which is a large-scalepretrained BERT model on Chinese texts. 

The detailed introduction of the model in this research is contained in CoCoGPT.pdf.

## Result

Human evaluation and automatic evaluation results show that these models are promising in generating clinically meaningful and linguistically high-quality consultations for COVID-19 in Chinese.The detailed performance of the model is contained in CoCoGPT.pdf.

The generated examples：

![image](https://github.com/kaamava/CoCoGPT/assets/106901273/53005c59-b492-4d65-9508-91e785412e26)
