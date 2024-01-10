import torch.utils.data

from transit_eeg.augmentations.feature_extractor import EEG_Net_8_Stack
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from src.utils import gather
import math

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    
class ArcMarginHead(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, load_backbone = './assets/max_acc.pth', s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginHead, self).__init__()
        self.arcpro = ArcMarginProduct(in_features, out_features, s=30.0, m=0.50, easy_margin=False)
        self.auxback = EEG_Net_8_Stack(mtl=False)
        pretrained_checkpoint = torch.load(load_backbone)
        print("loading the pretrained subject EEGNet weight and convert to arc..{}".format(pretrained_checkpoint.keys()))
        self.auxback.load_state_dict(pretrained_checkpoint['params'])
        print("backbone loading successful")

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        emb = self.auxback(input)
        # print(output)
        output = self.arcpro(emb, label)

        return output
    
class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, sub_theta: nn.Module, sub_arc_head: nn.Module, debug=False, time_diff_constraint=True):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head
        self.time_diff_constraint = time_diff_constraint

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        self.debug = debug
        self.step_size = 75
        self.window_size = 224
        # self.step_size = 93.5
        # self.window_size = 93.5
        self.subject_noise_range = 9
        # self.arcmargin = ArcMarginProduct()

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        if self.debug:
            print("the selected alpha bar would be {}".format(gather(self.alpha_bar, t).shape))
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps


    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def p_sample_noise(self, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.sub_theta(xt, t, s)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps


    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, debug=False):
        """
        #### Simplified Loss

        $$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        if debug:
            print("the shape of x0")
            print(x0.shape)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if debug:
            print("the shape of t")
            print(t.shape)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)
            if debug:
                print("the shape of noise")
                print(noise.shape)
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if debug:
            print("the shape of xt")
            print(xt.shape)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)
        # MSE loss
        return F.mse_loss(noise, eps_theta)

    
    def loss_with_diff_constraint(self, x0: torch.Tensor, label: torch.Tensor, 
                                    noise: Optional[torch.Tensor] = None, debug=False, 
                                    noise_content_kl_co = 1, arc_subject_co = 0.1, orgth_co = 2):
        """
        #### Simplified Loss

        $$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        debug = self.debug
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        if debug:
            print("the shape of x0")
            print(x0.shape)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        # s = torch.randint(0, self.subject_noise_range, (batch_size,), device=x0.device, dtype=torch.long)
        s = label
        if debug:
            print("the shape of t")
            print(t.shape)
            print(t)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)
            if debug:
                print("the shape of noise")
                print(noise.shape)
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if debug:
            print("the shape of xt")
            print(xt.shape)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)
        subject_mu, subject_theta = self.sub_theta(xt, t, s)
        # print(subject_theta.shape)
        if debug:
            print("the shape of eps_theta")
            print(eps_theta.shape)
            print("the shape of subject_theta")
            print(subject_theta.shape)
        constraint_panelty = 0
        
        # eps_layernorm = F.layer_norm(eps_theta, [eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[3]])
        for i in range(eps_theta.shape[3] - 1):
            # i * self.step_size, i * self.step_size + self.window_size
            # (i + 1) * self.step_size,  (i + 1) * self.step_size + self.window_size
            if debug:
                print("logging with constraint panelty value")
                # print(F.mse_loss(eps_layernorm[:, :, (i + 1) * self.step_size : i * self.step_size + self.window_size, i], eps_layernorm[:, :, (i + 1) * self.step_size : i * self.step_size + self.window_size, i + 1], reduction='none').shape)
                # print(F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='none').shape)
                # print(F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='mean'))
                print(F.mse_loss(eps_theta[:, :, self.step_size:, i], eps_theta[:, :, :-self.step_size, i + 1], reduction='mean'))
            # constraint_panelty = constraint_panelty +F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='mean')
            constraint_panelty = constraint_panelty +F.mse_loss(eps_theta[:, :, self.step_size:, i], eps_theta[:, :, :-self.step_size, i + 1], reduction='mean')
        if debug:
            print("logging with constraint panelty value")
            print(constraint_panelty)

        # MSE loss
        # return F.mse_loss(noise, eps_theta + subject_theta) + constraint_panelty, constraint_panelty
        # noise_conent_kl = F.kl_div(eps_theta.softmax(dim=-1).log(), subject_theta.softmax(dim=-1), reduction='sum')
        noise_conent_kl = F.kl_div(eps_theta.softmax(dim=-1).log(), subject_theta.softmax(dim=-1), reduction='mean')
        organal_squad = torch.bmm(eps_theta.view(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2],eps_theta.shape[3]), subject_theta.view(subject_theta.shape[0]*subject_theta.shape[1], subject_theta.shape[2], subject_theta.shape[3]).permute(0,2,1))
        if debug:
            print("logging with organal_squad")
            print(organal_squad.shape)
        ones = torch.ones(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[2], dtype=torch.float32, device='cuda') # (N * C) * H * H   
        diag = torch.eye(eps_theta.shape[2], dtype=torch.float32,device='cuda') # (N * C) * H * H
        loss_orth = ((organal_squad * (ones - diag).to('cuda')) ** 2).mean()
        if debug:
            print("logging with loss_orth")
            print(loss_orth)
        subject_arc_logit = self.sub_arc_head(subject_theta.permute(0,3,2,1), s)
        subject_arc_loss = F.cross_entropy(subject_arc_logit, s.long())
        # return F.mse_loss(noise, eps_theta + subject_theta) + 0.01 * 1/noise_conent_kl, constraint_panelty, noise_conent_kl
        # return F.mse_loss(noise, eps_theta + subject_theta) - noise_content_kl_co * noise_conent_kl + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss
        if self.time_diff_constraint:
            return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss + 0.1 * constraint_panelty, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth
        else:
            return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth
