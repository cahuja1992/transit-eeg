# transit_eeg/subject_identifier/eegnet.py

import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, in_ch=62, out_dim=200):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = out_dim
        self.proj = nn.Linear(16, out_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, C, F]
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 16]
        return self.proj(x)  # [B, out_dim]
