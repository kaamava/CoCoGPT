import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

# uses bert chinese wordpiece tokenization
from pytorch_pretrained_bert import OpenAIAdam


def train_model(
    epochs=10,
    num_gradients_accumulation=4,
    batch_size=4,
    gpu_id=0,
    lr=1e-5,
    load_dir='decoder_model'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    encoder = TransformerEncoder()
    decoder = TransformerDecoderLM()

    encoder.load_state_dict(torch.load("encoder.pth"))
    decoder.load_state_dict(torch.load("decoder.pth"))

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    train_data = torch.load("../train_data.pth")
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_data = torch.load("../validate_data.pth")
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------
    

    #------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(decoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=lr,
                        warmup=0.01,
                        max_grad_norm=1.0,
                        weight_decay=0.01,
                        t_total=num_train_optimization_steps)
    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        decoder.train()
        losses = 0
        times = 0
        for batch in train_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            _, past = encoder(encoder_input, mask_encoder_input)
        
            mask = torch.cat([mask_encoder_input, mask_decoder_input], dim=1)
            logits, _ = decoder(decoder_input, mask, past=past, past_length=0)
            
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()
            times += 1
            
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                optimizer.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        start = end

        #------------------------validate------------------------
        decoder.eval()

        perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch in val_dataloader:
                batch = [item.to(device) for item in batch]

                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

                _, past = encoder(encoder_input, mask_encoder_input)
            
                mask = torch.cat([mask_encoder_input, mask_decoder_input], dim=1)
                logits, _ = decoder(decoder_input, mask, past=past, past_length=0)
                
                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
                perplexity += np.exp(loss.item())
                batch_count += 1

        print(f'validate perplexity: {perplexity / batch_count}')

        torch.save(decoder.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "decoder.pth"))

    #------------------------END TRAINING-------------------


if __name__ == '__main__':
    fire.Fire(train_model)
