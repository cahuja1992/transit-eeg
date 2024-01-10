from multiprocessing import reduction
from turtle import Turtle
from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from src.EEGNet import EEG_Net_8_Stack
from src.unet_eeg_subject_emb import sub_gaussion
from src.unet_eeg import UNet
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from src.utils import gather
import numpy as np
import scipy.io as sio
import csv
import math

import argparse
import csv
import torch
from torch.utils.data import DataLoader
from unet import UNet  
from .embeddings import SubjectUNet 
from .ddpm import ArcMarginHead, DenoiseDiffusion 
from tqdm import tqdm
import wandb

class IDMP:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.eps_model = UNet()  # Initialize your UNet class
        self.sub_theta = sub_gaussion()  # Initialize your sub_gaussion class
        self.sub_archead = ArcMarginHead()  # Initialize your ArcMarginHead class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=args.n_steps,
            device=self.device,
            sub_theta=self.sub_theta,
            sub_arc_head=self.sub_archead,
            debug=False,
        )

        self.eeg_channels = args.eeg_channels
        self.window_size = args.window_size
        self.stack_size = args.stack_size
        self.n_channels = args.n_channels
        self.channel_multipliers = args.channel_multipliers
        self.is_attention = args.is_attention

        self.n_steps = args.n_steps
        self.batch_size = args.batch_size
        self.n_samples = args.n_samples
        self.learning_rate = args.learning_rate
        self.arc_in = args.arc_in
        self.arc_out = args.arc_out

        self.epochs = args.epochs

        self.dataset = YourDatasetClass()  # Initialize your dataset class
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.Adam([
            {'params': self.eps_model.parameters(), 'lr': self.learning_rate},
            {'params': self.sub_theta.parameters(), 'lr': self.learning_rate},
            {'params': self.sub_archead.arcpro.parameters(), 'lr': self.learning_rate}
        ], lr=self.learning_rate)
        self.optimizer_noise = torch.optim.Adam(self.sub_theta.parameters(), lr=self.learning_rate)

        # Initialize WandB
        wandb.init(project='your_project_name', config=vars(args))  # Set your WandB project name

    def sample(self):
        """
        Generate samples using the trained model and log to WandB.
        """
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.eeg_channels, self.window_size, self.stack_size], device=self.device)

            for t_ in tqdm(range(self.n_steps), desc='Sample'):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

    def train(self):
        """
        Train the model, log metrics to WandB, and write to a CSV file.
        """
        metric_file = 'metrics.csv'
        with open(metric_file, 'a+') as loss_target:
            target_writer = csv.writer(loss_target)
            for data, label in tqdm(self.data_loader, desc='Train'):
                self.optimizer.zero_grad()
                data = data.permute(0, 3, 2, 1).to(self.device)
                label = label.float().to(self.device)
                loss, time_period_diff, noise_conent_kl, sub_arc_loss, loss_orth = self.diffusion.loss_with_diff_constraint(data, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer_noise.step()

                # Log metrics to WandB
                wandb.log({'loss': loss, 'time_seg_diff': time_period_diff, 'noise_conent_kl': noise_conent_kl,
                           'sub_arc_loss': sub_arc_loss, 'loss_orth': loss_orth})

                # Write metrics to CSV
                target_writer.writerow([loss.detach().cpu().numpy(), (loss - loss_orth - sub_arc_loss).detach().cpu().numpy(),
                                       time_period_diff.detach().cpu().numpy(), noise_conent_kl.detach().cpu().numpy(),
                                       sub_arc_loss.detach().cpu().numpy(), loss_orth.detach().cpu().numpy()])

    def run(self):
        """
        Run the training loop with specified epochs.
        """
        for epoch in tqdm(range(self.epochs), desc='Training loop'):
            print(epoch)
            self.train()
            self.sample()


def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--device', default='cuda', help='Device to train the model on')
    parser.add_argument('--n_steps', type=int, default=1_000, help='Number of time steps T')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--arc_in', type=int, default=4*2*14, help='ArcMarginHead input size')
    parser.add_argument('--arc_out', type=int, default=9, help='ArcMarginHead output size')
    parser.add_argument('--epochs', type=int, default=1_000, help='Number of training epochs')
    # ... (add more arguments as needed)

    args = parser.parse_args()
    config = Configs(args)
    config.run()


if __name__ == "__main__":
    main()
