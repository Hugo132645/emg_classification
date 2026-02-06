import torch
import torch.nn as nn

from typing import Literal, Optional

#GRU model class with PyTorch
class GRUModel(nn.Module):
    #Initialization of hyperparameters
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_classes: int,
            num_layers: int = 2,
            bidirectional: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(output_dim, num_classes)

    #Defining forward pass
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, hidden = self.gru(x)

        if lengths is None:
            last = out[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)
            B = x.size(0)
            last = out[torch.arange(B), idx, :]

        logits = self.fc(last)
        return logits

#LSTM model class with PyTorch
class LSTMModel(nn.Module):
    #Initialization of hyperparameters
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_classes: int,
            num_layers: int = 2,
            bidirectional: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(output_dim, num_classes)

    #Forward pass for LSTM
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, hidden = self.lstm(x)

        if lengths is None:
            last = out[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)
            B = x.size(0)
            last = out[torch.arange(B), idx, :]

        logits = self.fc(last)
        return logits


