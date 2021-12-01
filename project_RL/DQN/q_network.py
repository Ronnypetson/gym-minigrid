import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        h: int,
        w: int
        ):
        assert h > 0 and w > 0, 'h and w must be positive integers.'
        super().__init__()
        self._conv = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU()
        )
        hidden_dim = 24 * h * w
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(
        self,
        x: torch.Tensor
        ):
        ''' x has shape (N, 1, H, W)
        '''
        x = self._conv(x)
        x = x.reshape(x.size(0), -1)
        x = self._mlp(x)
        return x
