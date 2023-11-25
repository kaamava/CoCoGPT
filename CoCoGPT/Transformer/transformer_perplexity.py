import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers_model import transformers_model

import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util


def calculate(
    batch_size=1,
    gpu_id=0,
    decoder_path='decoder_model'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transformers_model()
    model.load_state_dict(torch.load(decoder_path))
    device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    train_data = torch.load("../train_data.pth")
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    val_data = torch.load("../validate_data.pth")
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    test_data = torch.load("../test_data.pth")
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------


    #------------------------START TRAINING-------------------

    print('start training cal...')
    #------------------------training------------------------
    perplexity = 0
    batch_count = 0
    with torch.no_grad():
        for batch in train_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'train perplexity: {perplexity / batch_count}')

    #------------------------validate------------------------

    perplexity = 0
    batch_count = 0
    print('start calculate the perplexity....')

    with torch.no_grad():
        for batch in val_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'validate perplexity: {perplexity / batch_count}')


    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        for batch in test_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

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

