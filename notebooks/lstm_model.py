import torch
import torch.nn as nn
import torch.nn.functional as F

# define class for LSTM model
class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) based neural network model for sequence prediction tasks.
    Attributes:
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        lstm (nn.LSTM): LSTM layer for processing input sequences.
        fc (nn.Linear): Fully connected layer for output prediction.
        dropout (nn.Dropout): Dropout layer for regularization.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    Methods:
        __init__(input_size, hidden_size, num_layers, output_size):
            Initializes the LSTMModel with the given parameters.
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_size).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.view(-1)
    