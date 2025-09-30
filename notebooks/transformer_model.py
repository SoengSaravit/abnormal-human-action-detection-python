import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=120):
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
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        # x = x[:, -1, :]
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
