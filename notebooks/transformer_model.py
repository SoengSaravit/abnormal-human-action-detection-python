import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StaticPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(StaticPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_embeddings = self.pos_embed(positions)
        return x + pos_embeddings
    

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, tl_dropout=0.1, nn_dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # self.positional_encoding = StaticPositionalEncoding(d_model)
        
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=tl_dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(nn_dropout)

    def forward(self, x):
        # x_static = x[:, 1:, :]
        # x_diff = x[:, 1:, :] - x[:, :-1, :]
        # x = torch.cat([x_static, 3 * x_diff], axis=-1)
        # x = F.normalize(x, dim=-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = x.mean(dim=1)  # temporal average pooling
        x = self.fc(x)
        return x
