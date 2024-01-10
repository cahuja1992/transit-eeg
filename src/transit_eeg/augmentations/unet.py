import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

class TimeEmbedding(nn.Module):
    """
    Embeddings for time (t).
    """

    def __init__(self, n_channels: int):
        """
        Initialize TimeEmbedding.

        Parameters:
        - `n_channels`: Number of dimensions in the embedding.
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        Forward pass for TimeEmbedding.

        Parameters:
        - `t`: Tensor with shape [batch_size, time_channels].

        Returns:
        - Tensor with shape [batch_size, time_channels, n_channels].
        """
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class SubjectEmbedding(nn.Module):
    """
    Embeddings for subject (s).
    """

    def __init__(self, n_channels: int):
        """
        Initialize SubjectEmbedding.

        Parameters:
        - `n_channels`: Number of dimensions in the embedding.
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor, debug=False):
        """
        Forward pass for SubjectEmbedding.

        Parameters:
        - `t`: Tensor with shape [batch_size, time_channels].
        - `debug`: Debug flag.

        Returns:
        - Tensor with shape [batch_size, time_channels, n_channels].
        """
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """
    Residual block with two convolution layers.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        Initialize ResidualBlock.

        Parameters:
        - `in_channels`: Number of input channels.
        - `out_channels`: Number of output channels.
        - `time_channels`: Number of channels in the time step (t) embeddings.
        - `n_groups`: Number of groups for group normalization.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, debug=False):
        """
        Forward pass for ResidualBlock.

        Parameters:
        - `x`: Tensor with shape [batch_size, in_channels, height, width].
        - `t`: Tensor with shape [batch_size, time_channels].
        - `debug`: Debug flag.

        Returns:
        - Tensor with shape [batch_size, out_channels, height, width].
        """
        if debug:
            print('shape of it after norm1 is {}'.format(self.norm1(x).shape))
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Attention block similar to transformer multi-head attention.
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        Initialize AttentionBlock.

        Parameters:
        - `n_channels`: Number of channels in the input.
        - `n_heads`: Number of heads in multi-head attention.
        - `d_k`: Number of dimensions in each head.
        - `n_groups`: Number of groups for group normalization.
        """
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Forward pass for AttentionBlock.

        Parameters:
        - `x`: Tensor with shape [batch_size, in_channels, height, width].
        - `t`: Tensor with shape [batch_size, time_channels].

        Returns:
        - Tensor with shape [batch_size, out_channels, height, width].
        """
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block similar to transformer multi-head attention.
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        Initialize CrossAttentionBlock.

        Parameters:
        - `n_channels`: Number of channels in the input.
        - `n_heads`: Number of heads in multi-head attention.
        - `d_k`: Number of dimensions in each head.
        - `n_groups`: Number of groups for group normalization.
        """
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 2)
        self.query_projection = nn.Linear(n_channels, n_heads * d_k * 448)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, s: torch.Tensor, debug=False):
        """
        Forward pass for CrossAttentionBlock.

        Parameters:
        - `x`: Tensor with shape [batch_size, in_channels, height, width].
        - `s`: Tensor with shape [batch_size, subject_channels].
        - `debug`: Debug flag.

        Returns:
        - Tensor with shape [batch_size, out_channels, height, width].
        """
        _ = s
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        s = s.view(batch_size, n_channels, -1).permute(0, 2, 1)
        if debug:
            print('shape of x is :{}'.format(x.shape))
            print('shape of s is :{}'.format(s.shape))

        kv = self.projection(x).view(batch_size, -1, self.n_heads, 2 * self.d_k)
        k, v = torch.chunk(kv, 2, dim=-1)
        q = self.query_projection(s).view(batch_size, -1, self.n_heads, self.d_k)
        if debug:
            print('shape of q is :{}'.format(q.shape))
            print('shape of k is :{}'.format(k.shape))
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res

class DownBlock(nn.Module):
    """
    Down block combining ResidualBlock and AttentionBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class DownBlockSubjectFusion(nn.Module):
    """
    Down block combining ResidualBlock and CrossAttentionBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, 32)
        self.attn = CrossAttentionBlock(out_channels, n_groups=32) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x, s)
        return x

class UpBlockSubjectFusion(nn.Module):
    """
    Up block combining ResidualBlock and CrossAttentionBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, 32)
        self.attn = CrossAttentionBlock(out_channels, n_groups=32) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x, s)
        return x


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class BottleNeckBlock(nn.Module):
    """
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x

class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)

class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)

class SubjectUNet(nn.Module):

    def __init__(
        self, eeg_channels: int = 3, 
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2
    ):
        """
        * `eeg_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.mu_projection = nn.Conv2d(eeg_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.theta_projection = nn.Conv2d(eeg_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)
        self.sub_emb = SubjectEmbedding(n_channels * 4)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.down_fc = DownBlockSubjectFusion(n_channels, n_channels, n_channels * 4, True)
        self.up_fc = UpBlockSubjectFusion(n_channels, n_channels, n_channels * 4, True)
        self.down_fc_mu = DownBlockSubjectFusion(n_channels, n_channels, n_channels * 4, True)
        self.up_fc_mu = UpBlockSubjectFusion(n_channels, n_channels, n_channels * 4, True)
        self.final_mu = nn.Conv2d(n_channels, eeg_channels, kernel_size=(3, 3), padding=(1, 1))
        self.final_theta = nn.Conv2d(n_channels, eeg_channels, kernel_size=(3, 3), padding=(1, 1))


    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, debug=False):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)
        sub = self.sub_emb(s)
        if debug:
            print('the shape of the time emb is {}'.format(t.shape))
            print('the shape of the subject emb is {}'.format(sub.shape))

        # Get image projection
        mu = self.mu_projection(x)
        theta = self.theta_projection(x)
        if debug:
            print('the shape after mu projection is {}'.format(mu.shape))
            print('the shape after theta projection is {}'.format(theta.shape))
        

        theta = self.down_fc(theta, t, sub)
        if debug:
            print('the shape after mu downsampe is {}'.format(mu.shape))
        theta = self.up_fc(theta, t, sub)
        
        mu = self.down_fc_mu(mu, t, sub)
        mu = self.up_fc_mu(mu, t, sub)
        if debug:
            print('the shape after mu down up is {}'.format(mu.shape))
            print('the shape after mu down up is {}'.format(theta.shape))
        
        # `h` will store outputs at each resolution for skip connectio
        # Final normalization and convolution
        return self.final_mu(self.act(self.norm(mu))), self.final_theta(self.act(self.norm(theta)))


class UNet(nn.Module):
    def __init__(
        self, eeg_channels: int = 3, 
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2
    ):
        """
        * `eeg_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(eeg_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Bottlenect block
        self.bottleneck = BottleNeckBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, eeg_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, debug=False):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)
            
        # Bottleneck (bottom)
        x = self.bottleneck(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))