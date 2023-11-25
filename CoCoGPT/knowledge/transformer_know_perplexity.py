import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, BertModel, BertConfig, get_linear_schedule_with_warmup

import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

hidden_size = 256
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
#        print (knowledge_bias.size(), len(knowledge_bias.size()))
        if len(knowledge_bias.size()) == 1:
            knowledge_bias = knowledge_bias.unsqueeze(0)
#            print ("enter")
#        print (knowledge_bias.size(), len(knowledge_bias.size()))
        knowledge_bias = knowledge_bias.expand(out.size()[1], knowledge_bias.size()[0], knowledge_bias.size()[1]).permute(1, 0, 2)
#        print (out.size())
#        print (knowledge_bias.size())
        out = out + knowledge_bias
        return out

def calculate(
    batch_size=1,
    gpu_id=0,
    decoder_path='./decoder_model/49model.pth'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transforers_model()
    model.load_state_dict(torch.load(decoder_path))
    device = torch.device(f"cuda:7")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
#    train_data = torch.load("train_data.pth")
#    train_dataset = TensorDataset(*train_data)
#    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
#    val_data = torch.load("validate_data.pth")
#    val_dataset = TensorDataset(*val_data)
#    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    
    knowledge = torch.load("knowledge.pth")
    knowledge_size = knowledge.size()
#    print (knowledge_size)
    
    test_data = torch.load("test_data.pth")
    test_knowledge = knowledge.expand(len(test_data[0]), knowledge_size[0])
    
    test_dataset = TensorDataset(*test_data, test_knowledge)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------


    #------------------------START TRAINING-------------------
#
#    print('start training cal...')
#    #------------------------training------------------------
#    perplexity = 0
#    batch_count = 0
#    with torch.no_grad():
#        for batch in train_dataloader:
#            batch = [item.to(device) for item in batch]
#
#            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
#            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
#
#            out = logits[:, :-1].contiguous()
#            target = decoder_input[:, 1:].contiguous()
#            target_mask = mask_decoder_input[:, 1:].contiguous()
#
#            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
#
#            perplexity += np.exp(loss.item())
#
#            batch_count += 1
#
#    print(f'train perplexity: {perplexity / batch_count}')
#
#    #------------------------validate------------------------
#
#    perplexity = 0
#    batch_count = 0
#    print('start calculate the perplexity....')
#
#    with torch.no_grad():
#        for batch in val_dataloader:
#            batch = [item.to(device) for item in batch]
#
#            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
#            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
#
#            out = logits[:, :-1].contiguous()
#            target = decoder_input[:, 1:].contiguous()
#            target_mask = mask_decoder_input[:, 1:].contiguous()
#
#            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
#
#            perplexity += np.exp(loss.item())
#
#            batch_count += 1
#
#    print(f'validate perplexity: {perplexity / batch_count}')


    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        for batch in test_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, knowledge_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'test perplexity: {perplexity / batch_count}')

    #------------------------END cal-------------------


if __name__ == '__main__':
    fire.Fire(calculate)
