"""Models for P21.

Baseline          : RandomForest on the raw 187 samples (sklearn)
Main model        : LSTM and BiLSTM on the beat as a length-187 sequence
Improvement       : CNN-LSTM (1-D conv front end -> LSTM) + class-weighted loss
                    (the weighting is switched on in train.py with --weighted,
                    so the two ingredients can be ablated separately)

torch is imported lazily so the sklearn parts work without it.
"""
from sklearn.ensemble import RandomForestClassifier

try:
    import torch.nn as nn
except ImportError:  # sklearn-only environments
    nn = None

_Module = nn.Module if nn is not None else object


def build_rf(seed, n_estimators=200):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)


class LSTMClassifier(_Module):
    def __init__(self, hidden=64, layers=1, bidirectional=False, n_classes=5, dropout=0.2):
        super().__init__()
        self.bi = bidirectional
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):                     # x: (B, 187)
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        h = torch_cat_last(h, self.bi)
        return self.fc(self.drop(h))


class CNNLSTM(_Module):
    def __init__(self, hidden=64, bidirectional=False, n_classes=5, dropout=0.2):
        super().__init__()
        self.bi = bidirectional
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )                                      # 187 -> 93 -> 46 time steps
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)   # (B, T, C)
        _, (h, _) = self.lstm(z)
        h = torch_cat_last(h, self.bi)
        return self.fc(self.drop(h))


def torch_cat_last(h, bidirectional):
    """Final hidden state of the last layer (concatenate both directions if bi)."""
    import torch
    return torch.cat([h[-2], h[-1]], dim=1) if bidirectional else h[-1]


def build_torch_model(name):
    if name == "lstm":
        return LSTMClassifier(bidirectional=False)
    if name == "bilstm":
        return LSTMClassifier(bidirectional=True)
    if name == "cnnlstm":
        return CNNLSTM()
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
