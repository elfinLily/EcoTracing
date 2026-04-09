import torch
import torch.nn as nn

class EnergyMLP(nn.Module):
    """
    에너지 예측 MLP 모델
    - hidden_sizes : 각 은닉층 뉴런 수 리스트
    - dropout_rate : Dropout 비율
    """
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.0, use_batchnorm=True):
        super(EnergyMLP, self).__init__()
        layers = []
        in_size = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_size = h

        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)