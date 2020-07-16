import torch
from torch import nn

class LSTMModel(nn.Module):
    
    def __init__(self, history_size=1, hidden_size=100):
        
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=history_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.cls = nn.Linear(in_features=hidden_size, out_features=1)
        
    def forward(self, x):

        output, _ = self.lstm(x)
        return self.cls(output).view(x.shape[1], -1)[-1]