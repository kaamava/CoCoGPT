import torch.nn as nn
from transformers import BertModel, BertConfig


class transformers_model(nn.Module):
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

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        # out: [batch_size, max_length, hidden_size]
        out, _ = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        out = self.linear(out)
        return out
