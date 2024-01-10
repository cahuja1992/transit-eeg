
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Linear, Dropout, Conv2d, MaxPool2d
from torch_geometric.utils import to_dense_batch

class SOGAT(nn.Module):
    """
    Self-Organizing Graph Attention Network.

    Args:
        None
    """
    def __init__(self):
        super().__init__()

        drop_rate = 0.1
        topk = 10
        self.channels = 62

        self.conv1 = Conv2d(1, 32, (5, 5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, 4))
        self.sogc1 = SOGC(65 * 32, 64, 32, topk)

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1, 4))
        self.sogc2 = SOGC(15 * 64, 64, 32, topk)

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1, 4))
        self.sogc3 = SOGC(2 * 128, 64, 32, topk)

        self.drop4 = Dropout(drop_rate)

        self.linend = Linear(self.channels * 32 * 3, 3)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the Self-Organizing Graph Attention Network.

        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Edge index tensor.
            batch (torch.Tensor): Batch tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and predicted probabilities.
        """
        x, mask = to_dense_batch(x, batch)

        x = x.reshape(-1, 1, 5, 265)  # (Batch*channels, 1, Freq_bands, Features)

        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x1 = self.sogc1(x)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x2 = self.sogc2(x)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)

        x3 = self.sogc3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.drop4(x)

        x = x.reshape(-1, self.channels * 32 * 3)
        x = self.linend(x)
        pred = F.softmax(x, 1)

        return x, pred

