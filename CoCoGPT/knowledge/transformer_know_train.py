import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, BertModel, BertConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

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

def matrix_mul(input, weight, bias=False):
    feature_list = []
#    print ("input: ", input)
    for feature in input:
#        print ("feature: ", feature)
#        print ("weight: ", weight)
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0)

class SentAttNet(nn.Module):
    def __init__(self):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.gru = nn.GRU(2 * hidden_size, hidden_size, bidirectional=True)
#        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):

        sent_hiddens, last_hidden = self.gru(input)
        
        uit = matrix_mul(sent_hiddens, self.sent_weight, self.sent_bias)
        alpha = matrix_mul(uit, self.context_weight).permute(1, 0)
        alpha = F.softmax(alpha)
        output = element_wise_mul(sent_hiddens, alpha.permute(1, 0))
#        output = self.fc(output)

        return output

class WordAttNet(nn.Module):
    def __init__(self):
        super(WordAttNet, self).__init__()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):
        
        word_hidden, last_hidden = self.gru(input)
#        print (last_hidden)
        #len * batch_size * (2 * hidden_size)
#        print (word_hidden.size())
        
        uit = matrix_mul(word_hidden, self.word_weight, self.word_bias)
        alpha = matrix_mul(uit, self.context_weight).permute(1, 0)
        # len * batch_size ->  batch_size * len
        alpha = F.softmax(alpha)
        output = element_wise_mul(word_hidden, alpha.permute(1, 0))
#        print (output.size())
#        output = torch.mean(output, dim = 1)

        return output

class DocClass(nn.Module):
    def __init__(self):
        super(DocClass, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_size, padding_idx = 0)
        self.sent_hidden = WordAttNet()
        self.docu_hidden = SentAttNet()
        self.classification = nn.Linear(2 * hidden_size, num_classes)
    
    def forward(self, input):
        input = input.permute(1, 0, 2)
        sent_list = []
        for sent in input:
            word_embedding = self.word_embedding(sent).permute(1, 0, 2)
            sent_hidden = self.sent_hidden(word_embedding).unsqueeze(0)
            sent_list.append(sent_hidden)
        sents = torch.cat(sent_list, 0)
        output = self.docu_hidden(sents)
        
        result = self.classification(output)
#        print (result.size())
        return result

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



def train_model(
    epochs=50,
    num_gradients_accumulation=4,
    batch_size=4,
    gpu_id=0,
    lr=1e-5,
    load_dir='decoder_model'
    ):
    # make sure your model is on GPU
    device = torch.device("cuda:6,7")

    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transforers_model()
#    device = torch.device("cuda:0")
    model.to(device)

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    knowledge = torch.load("knowledge.pth")
    knowledge_size = knowledge.size()
    
    train_data = torch.load("train_data.pth")
    train_knowledge = knowledge.expand(len(train_data[0]), knowledge_size[0])
#    print (train_knowledge.size())
    train_dataset = TensorDataset(*train_data, train_knowledge)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    
    val_data = torch.load("validate_data.pth")
    val_knowledge = knowledge.expand(len(val_data[0]), knowledge_size[0])
    print (val_knowledge.size())
    val_dataset = TensorDataset(*val_data, val_knowledge)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------
    

    #------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,\
        lr=lr,\
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10, \
        num_training_steps=num_train_optimization_steps
    )
    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        for batch in tqdm(train_dataloader):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, knowledge_input = batch
#            print (knowledge_input.size())
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()

            times += 1
            update_count += 1
            max_grad_norm = 1.0
            
            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + 'epoch' + str(epoch) + '-'*20)
        print('time:' + str(end - start))
        print('loss:' + str(losses / times))
        start = end

        #------------------------validate------------------------
        model.eval()

        perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = [item.to(device) for item in batch]

                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, knowledge_input = batch
                logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input)

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

                perplexity += np.exp(loss.item())

                batch_count += 1

        print('validate perplexity:' + str(perplexity / batch_count))
        
        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)

        torch.save(model.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "model.pth"))

    #------------------------END TRAINING-------------------


if __name__ == '__main__':
    fire.Fire(train_model)

