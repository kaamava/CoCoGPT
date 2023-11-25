# CoCoGPT

## Introduction

Today, with the increasing risk of COVID-19 infection in the society, people who are suspected of COVID-19's symptoms or have a greater risk of infection for various reasons will have some anxiety about medical consultation. Relatively, the pressure on the medical system will rapidly increase in the short term. Opening up an online medical consultation platform can to some extent alleviate the current social pressure and public anxiety. However, at the same time, the professionalism of online consultation needs to be guaranteed. Therefore, the purpose of this study is to establish a medical dialogue model CoCoGPT (COVID-19 Consultant GPT) that can provide COVID-19 related consultations using professional medical consultation materials. The report utilizes and improves a Chinese medical dialogue dataset COVID-Dialogue-Chinese, which includes dialogue records between doctors and patients regarding COVID-19. In this study, the dataset were used to train a BERT-GPT based model.

# Dataset
COVID-Dialogue-Dataset-Chinese is a Chinese medical dialogue dataset about COVID-19 and other types of pneumonia. Patients who are concerned that they may be infected by COVID-19 or other pneumonia consult doctors and doctors provide advice. There are 1088 consultations. Each consultation consists of

- ID
- URL
- Description of patient’s medical condition
- Dialogue
- (Optional) Diagnosis and suggestions.

The dataset is built from [Haodf.com](https://www.haodf.com/) and all copyrights of the data belong to [Haodf.com](https://www.haodf.com/).

## Model




1. First preprocess the dataset and split the data to train, validate and test sets:

   ```shell
   $ python preprocess.py
   ```

2. Get into one of the models directory (take Bert-GPT as an example)

   ```shell
   $ cd Bert-GPT
   ```

3. Train the dialogue generation model:

   ``` shell
   $ python train.py --load_dir ${LOAD_DIR}
   ```

4. Evaluation the trained dialogue generation model:

   ```shell
   $ python calculate_perplexity.py --decoder_path ${DECODER_PATH}
   ```

5. Generate responses using the trained dialogue generation model:

   ```shell
   $ python sample_generate.py --decoder_path ${DECODER_PATH}
   ```
