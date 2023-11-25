import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import BertModel, BertConfig
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

embed_size = 512
embed_num = 21128
num_classes = 21128
max_len = 8750

class CNNClass(nn.Module):
    def __init__(self):
        super(CNNClass, self).__init__()
        
        self.word_embedding = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_size)
        self.cnn2 = nn.Conv2d(in_channels = 1, out_channels = embed_size // 4, kernel_size = (2, embed_size))
        self.cnn3 = nn.Conv2d(in_channels = 1, out_channels = embed_size // 4, kernel_size = (3, embed_size))
        self.cnn4 = nn.Conv2d(in_channels = 1, out_channels = embed_size // 4, kernel_size = (4, embed_size))
        self.cnn5 = nn.Conv2d(in_channels = 1, out_channels = embed_size // 4, kernel_size = (5, embed_size))
        self.pool2 = nn.MaxPool2d(kernel_size = (max_len + 1 - 2, 1))
        self.pool3 = nn.MaxPool2d(kernel_size = (max_len + 1 - 3, 1))
        self.pool4 = nn.MaxPool2d(kernel_size = (max_len + 1 - 4, 1))
        self.pool5 = nn.MaxPool2d(kernel_size = (max_len + 1 - 5, 1))
        self.transfer = nn.Linear(embed_size, num_classes)
        
    def forward(self, input):
        
        word_embedding = self.word_embedding(input)
        cnn_input = word_embedding.unsqueeze(0).permute(1, 0, 2, 3)
        
        cnn2 = self.cnn2(cnn_input)
        pool2 = self.pool2(cnn2)
        
        cnn3 = self.cnn3(cnn_input)
        pool3 = self.pool3(cnn3)
        
        cnn4 = self.cnn4(cnn_input)
        pool4 = self.pool4(cnn4)
        
        cnn5 = self.cnn5(cnn_input)
        pool5 = self.pool5(cnn5)
#        print ("pool.size: ", pool5.size())
        feature = torch.cat((pool2, pool3, pool4, pool5), 1).squeeze()
#        print ("feature.size: ", feature.size())
        result = self.transfer(feature)
        return result


class transforers_model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )
        self.encoder = BertModel(encoder_config)

        decoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )
        decoder_config.is_decoder = True
        self.decoder = BertModel(decoder_config)

        self.linear = nn.Linear(512, 21128, bias=False)
        self.get_knowledge = CNNClass()

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input, knowledge_input):
        encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        # out: [batch_size, max_length, hidden_size]
        out, _ = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        
        out = self.linear(out)
        knowledge_bias = self.get_knowledge(knowledge_input)
        knowledge_bias = knowledge_bias.expand(out.size()[1], knowledge_bias.size()[0], knowledge_bias.size()[1]).permute(1, 0, 2)
#        print (out.size())
#        print (knowledge_bias.size())
        out = out + knowledge_bias
        return out


def sample_generate(
    top_k = 50,
    temperature = 1.0,
    decoder_path='./decoder_model/49model.pth',
    gpu_id=0
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    model = transforers_model()
    model.load_state_dict(torch.load(decoder_path))

    device = torch.device(f"cuda:6,7")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD VALIDATE DATA------------------
    knowledge = torch.load("knowledge.pth")
    knowledge_size = knowledge.size()
    
    val_data = torch.load("test_data.pth")
    val_knowledge = knowledge.expand(len(val_data[0]), knowledge_size[0])
    
    val_dataset = TensorDataset(*val_data, val_knowledge)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
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
    f = open("dialogue.txt", "w")
    for batch in val_dataloader:
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _, knowledge_input = batch

            past, _ = model.encoder(encoder_input, mask_encoder_input)
            bias = model.get_knowledge(knowledge_input)
#            print (bias.size())
#            bias = bias.expand(100, bias.size()[0], bias.size()[1]).permute(1, 0, 2)
            
            
            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                logits, _ = model.decoder(sentence, encoder_hidden_states=past, )
                logits = model.linear(logits)
#                print (logits.size())
                
                logits = logits + bias
                
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
            f.write('-'*20 + f"example {update_count}" + '-'*20)
            f.write(f"input: {''.join(inputs)}")
            f.write(f"output: {''.join(reference)}")
            f.write(f"predict: {''.join(predict)}")

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

