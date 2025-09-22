import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=90):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, tl_dropout=0.1, nn_dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=tl_dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(nn_dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
