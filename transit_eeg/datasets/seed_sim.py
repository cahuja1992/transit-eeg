import numpy as np
import torch
from torch.utils.data import Dataset

class SimulatedSEED(Dataset):
    def __init__(self, num_subjects=6, samples_per_subject=30, channels=62, freqs=5, noise_scale=0.01):
        self.samples = []
        self.labels = []
        np.random.seed(42)

        # Generate orthogonal prototypes using QR decomposition
        orthogonal_basis = np.linalg.qr(np.random.randn(channels * freqs, channels * freqs))[0]
        subject_prototypes = orthogonal_basis[:num_subjects].reshape(num_subjects, channels, freqs)

        for i in range(num_subjects):
            for _ in range(samples_per_subject):
                noise = noise_scale * np.random.randn(channels, freqs)
                sample = subject_prototypes[i] + noise
                self.samples.append(sample)
                self.labels.append(i)

        self.samples = np.stack(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(1)  # shape: [C, 1, F]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
