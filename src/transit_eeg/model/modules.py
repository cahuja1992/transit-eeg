import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Linear, Dropout, Conv2d, MaxPool2d
from torch_geometric.utils import to_dense_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from typing import Optional


def glorot(tensor):
    """Glorot/Xavier uniform initialization."""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    """Zero initialization."""
    if tensor is not None:
        tensor.data.fill_(0)

class AdapterLayer(nn.Module):
    """
    Adapter Layer module that applies linear transformations to input data.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        output_channels (int): Number of output channels.
        non_linear_activation (torch.nn.Module, optional): Non-linear activation function. Defaults to None.
        bias (bool, optional): Whether to include bias in linear transformations. Defaults to False.
    """
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, non_linear_activation: nn.Module = None, bias: bool = False):
        super(AdapterLayer, self).__init__()

        self.linear1 = Linear(input_channels, hidden_channels, bias=bias)
        self.non_linear_activation = non_linear_activation
        self.linear2 = Linear(hidden_channels, output_channels, bias=bias)

    def forward(self, x):
        """
        Forward pass of the Adapter Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.linear1(x)
        if self.non_linear_activation is not None:
            x = self.non_linear_activation(x)
        x = self.linear2(x)
        return x


class LowRankAdapterLayer(nn.Module):
    """
    Low-Rank Adapter Layer module that decomposes linear transformations into low-rank components.

    Args:
        input_channels (int): Number of input channels.
        rank (int): Rank of the low-rank components.
        output_channels (int): Number of output channels.
        non_linear_activation (torch.nn.Module, optional): Non-linear activation function. Defaults to None.
        bias (bool, optional): Whether to include bias in linear transformations. Defaults to False.
    """
    def __init__(self, input_channels: int, rank: int, output_channels: int, non_linear_activation: nn.Module = None, bias: bool = False):
        super(LowRankAdapterLayer, self).__init__()

        self.rank = rank
        hidden_channels = min(input_channels, output_channels)

        # Decompose linear transformations into low-rank components
        self.lora_matrix_B = Parameter(torch.zeros(input_channels, rank))
        self.lora_matrix_A = Parameter(torch.randn(rank, input_channels))
        self.non_linear_activation = non_linear_activation

    def forward(self, x):
        """
        Forward pass of the Low-Rank Adapter Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        lora_weights = torch.matmul(self.lora_matrix_B, self.lora_matrix_A)
        return x + F.linear(x, lora_weights)


class DenseGATConv(nn.Module):
    """
    Dense Graph Attention Convolutional Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat (bool, optional): Whether to concatenate attention heads. Defaults to True.
        negative_slope (float, optional): Slope of the Leaky ReLU activation. Defaults to 0.2.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to include bias. Defaults to True.
        adapter (bool, optional): Whether to include adapter layers. Defaults to True.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, bias: bool = True, adapter: bool = True):
        super(DenseGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.adapter = adapter

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = Parameter(torch.empty(1, 1, heads, out_channels), requires_grad=True)
        self.att_dst = Parameter(torch.empty(1, 1, heads, out_channels), requires_grad=True)

        # Create Adapter Layers
        self.adapter_alpha_src = LowRankAdapterLayer(self.heads, self.heads, self.heads, non_linear_activation=torch.nn.ReLU(), bias=False)
        self.adapter_alpha_dst = LowRankAdapterLayer(self.heads, self.heads, self.heads, non_linear_activation=torch.nn.ReLU(), bias=False)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def freeze_layers(self, freeze=True):
        self.lin.requires_grad_(not freeze)
        self.att_src.requires_grad_(not freeze)
        self.att_dst.requires_grad_(not freeze)
        self.adapter_alpha_src.requires_grad_(freeze)
        self.adapter_alpha_dst.requires_grad_(freeze)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None, add_loop: bool = True):
        """
        Forward pass of the Dense GAT Convolutional Layer.

        Args:
            x (torch.Tensor): Input tensor.
            adj (torch.Tensor): Adjacency matrix.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            add_loop (bool, optional): Whether to add self-loops to the adjacency matrix. Defaults to True.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x = self.lin(x).view(B, N, H, C)

        alpha_src = torch.sum(x * self.att_src, dim=-1)
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)

        # Apply linear adapter layers
        if self.adapter:
            alpha_src = self.adapter_alpha_src(alpha_src)
            alpha_dst = self.adapter_alpha_dst(alpha_dst)

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SOGC(nn.Module):
    """
    Self-Organizing Graph Convolutional Layer.

    Args:
        in_features (int): Number of input features.
        bn_features (int): Number of batch normalization features.
        out_features (int): Number of output features.
        topk (int): Top-k value for sparsification.
    """
    def __init__(self, in_features: int, bn_features: int, out_features: int, topk: int):
        super().__init__()

        self.channels = 62
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk

        self.bnlin = Linear(in_features, bn_features)
        self.gconv = DenseGATConv(in_features, out_features, heads=1, dropout=0.0, adapter=False)

    def forward(self, x):
        """
        Forward pass of the Self-Organizing Graph Convolutional Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.reshape(-1, self.channels, self.in_features)
        xa = torch.tanh(self.bnlin(x))
        adj = torch.matmul(xa, xa.transpose(2, 1))
        adj = torch.softmax(adj, 2)
        amask = torch.zeros(xa.size(0), self.channels, self.channels).to(device)
        amask.fill_(0.0)
        s, t = adj.topk(self.topk, 2)
        amask.scatter_(2, t, s.fill_(1))
        adj = adj * amask
        x = F.relu(self.gconv(x, adj))

        return x

