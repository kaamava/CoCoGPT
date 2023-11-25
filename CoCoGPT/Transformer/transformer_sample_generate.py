import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers_model import transformers_model
from pytorch_pretrained_bert import BertTokenizer

import fire
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams

def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))


def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def cal_length(sentences):
    sen_length = [len(s.split()) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)


def calculate_metrics(predict, reference):
    reference_len = len(reference)
    predict_len = len(predict)

    #-------------------bleu----------
    bleu_2 = bleu(predict, reference, 2)
    bleu_4 = bleu(predict, reference, 4)
    #-------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    #-------------------meteor----------
    predict = " ".join(predict)
    reference = " ".join(reference)
    meteor_scores = meteor_score([reference], predict)
    return bleu_2, bleu_4, nist_2, nist_4, meteor_scores


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_generate(
    top_k = 50,
    temperature = 1.0,
    decoder_path='decoder.pth',
    gpu_id=0
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    model = transformers_model()
    model.load_state_dict(torch.load(decoder_path))

    device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD VALIDATE DATA------------------
    test_data = torch.load("../test_data.pth")
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    #------------------------END LOAD VALIDATE DATA--------------

    #------------------------START GENERETE-------------------
    update_count = 0

    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0
    
    meteor_scores = 0
    sentences = []
    print('start generating....')
    for batch in test_dataloader:
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            past, _ = model.encoder(encoder_input, mask_encoder_input)


            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                logits, _ = model.decoder(sentence, encoder_hidden_states=past)
                logits = model.linear(logits)
                logits = logits[:, -1]
                logits = logits.squeeze(1) / temperature
                
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence= torch.cat([sentence, prev_pred], dim=-1)
                if prev_pred[0][0] == 102:
                    break

            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()

            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())
            print('-'*20 + f"example {update_count}" + '-'*20)
            print(f"input: {''.join(inputs)}")
            print(f"output: {''.join(reference)}")
            print(f"predict: {''.join(predict)}")

            temp_bleu_2, \
            temp_bleu_4, \
            temp_nist_2, \
            temp_nist_4, \
            temp_meteor_scores = calculate_metrics(predict[1:-1], reference[1:-1])

            bleu_2scores += temp_bleu_2
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

    entro, dist = cal_entropy(sentences)
    mean_len, var_len = cal_length(sentences)
    print(f'avg: {mean_len}, var: {var_len}')
    print(f'entro: {entro}')
    print(f'dist: {dist}')
    print(f'test bleu_2scores: {bleu_2scores / update_count}')
    print(f'test bleu_4scores: {bleu_4scores / update_count}')
    print(f'test nist_2scores: {nist_2scores / update_count}')
    print(f'test nist_4scores: {nist_4scores / update_count}')
    print(f'test meteor_scores: {meteor_scores / update_count}')


if __name__ == '__main__':
    fire.Fire(sample_generate)

