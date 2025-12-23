# TRANSIT-EEG: Transfer and Robust Adaptation for New Subjects in EEG Technology

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](Transit_EEG_Publication.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **TRANSIT-EEG: A Framework for Cross-Subject Classification with Subject Specific Adaptation** by Chirag Ahuja and Divyashikha Sethia.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

TRANSIT-EEG addresses the challenge of cross-subject EEG signal classification by introducing a novel three-phase framework:

1. **Phase 1 - Pretraining**: Train IDPM (Individualised Diffusion Probabilistic Model) for subject-specific data augmentation and SOGAT (Self-Organizing Graph Attention Transformer) for classification
2. **Phase 2 - Finetuning**: Adapt to new subjects using LoRA (Low-Rank Adaptation) with minimal samples
3. **Phase 3 - Inference**: Deploy the fine-tuned model on unseen EEG sessions

### Key Achievements

- **SEED Dataset**: 91.53% F1-score for emotion recognition (3 classes: positive, negative, neutral)
- **PhyAat Dataset**: 87.78% F1-score for auditory activity recognition (3 classes: listening, writing, resting)
- **Few-shot Learning**: Adapts to new subjects with only 50% of one session
- **Synthetic Data Generation**: IDPM generates subject-specific augmented samples

## ✨ Key Features

### 1. IDPM (Individualised Diffusion Probabilistic Model)
- Decomposes EEG signals into clean components and subject-specific artifacts
- Generates high-quality synthetic samples for data augmentation
- Uses three loss functions: Reverse Loss, Orthogonal Loss, and Arc-Margin Loss
- Based on UNet architecture with dual-stream denoising

### 2. SOGAT (Self-Organizing Graph Attention Transformer)
- Dynamic graph construction for each subject based on input signals
- Graph Attention Network (GAT) for capturing inter-channel relationships
- Processes 5 frequency bands (theta, alpha, beta, gamma, delta) using Differential Entropy
- 386,991 trainable parameters optimized for efficiency

### 3. LoRA Adaptation
- Low-rank decomposition for efficient finetuning
- Prevents catastrophic forgetting
- Adapts source and destination attention matrices in GAT layers
- Minimal computational overhead

## 🏗️ Architecture

```
TRANSIT-EEG Framework
│
├── Phase 1: Pretraining
│   ├── IDPM Training (N-1 subjects)
│   │   ├── Forward Diffusion Process
│   │   ├── Reverse Diffusion Process
│   │   └── Loss Functions (Lr, Lo, Larc)
│   │
│   └── SOGAT Training
│       ├── Differential Entropy Feature Extraction
│       ├── Self-Organizing Graph Construction
│       └── Dense GAT Convolution Layers
│
├── Phase 2: Finetuning (New Subject)
│   ├── IDPM Sampling (Generate synthetic data)
│   └── LoRA Adapter Finetuning
│       ├── Low-Rank Matrices (A, B)
│       └── Freeze Base Weights
│
└── Phase 3: Inference
    └── Fine-tuned SOGAT Classification
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/cahuja1992/transit-eeg.git
cd transit-eeg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 📦 Quick Start

### 1. Download Datasets

```bash
# Download SEED dataset
python scripts/download_seed.py --output ./data/SEED

# Download PhyAat dataset
python scripts/download_phyaat.py --output ./data/PhyAat
```

### 2. Preprocess Data

```bash
# Extract Differential Entropy features
python scripts/preprocess.py \
    --dataset SEED \
    --input ./data/SEED/raw \
    --output ./data/SEED/processed
```

### 3. Train TRANSIT-EEG

```bash
# Phase 1: Pretrain IDPM + SOGAT
python train.py \
    --config configs/seed_pretrain.yaml \
    --dataset SEED \
    --output ./checkpoints/seed_pretrain

# Phase 2: Finetune with LoRA
python finetune.py \
    --config configs/seed_finetune.yaml \
    --checkpoint ./checkpoints/seed_pretrain/best_model.pt \
    --subject 0 \
    --output ./checkpoints/seed_finetune

# Phase 3: Evaluate
python evaluate.py \
    --checkpoint ./checkpoints/seed_finetune/subject_0.pt \
    --dataset SEED \
    --subject 0
```

### 4. Run LOSO (Leave-One-Subject-Out) Evaluation

```bash
# Complete LOSO evaluation on SEED
python scripts/run_loso.py \
    --config configs/seed_loso.yaml \
    --dataset SEED \
    --output ./results/seed_loso
```

## 📊 Datasets

### SEED (Emotional Recognition)

- **Source**: [SJTU BCMI Lab](http://bcmi.sjtu.edu.cn/~seed/)
- **Subjects**: 15 participants (7 male, 8 female)
- **Channels**: 62 EEG channels (ESI NeuroScan System)
- **Sampling Rate**: 1000 Hz (downsampled to 200 Hz)
- **Sessions**: 3 sessions per subject (45 trials each)
- **Classes**: 3 emotions (Positive=1, Negative=-1, Neutral=0)
- **Features**: 5 frequency bands × 62 channels × 265 features = 310 features/trial

### PhyAat (Auditory Activity Recognition)

- **Source**: [Physionet AAT Dataset](https://physionet.org/)
- **Subjects**: 25 healthy participants
- **Channels**: 14 EEG channels (Emotiv Epoc)
- **Sampling Rate**: 128 Hz
- **Classes**: 3 activities (Listening=0, Writing=1, Resting=2)
- **Features**: 5 frequency bands × 14 channels × 384 features = 70 features/trial

## 🎓 Training

### Configuration Files

All hyperparameters are specified in YAML configuration files:

```yaml
# configs/seed_pretrain.yaml
model:
  name: SOGAT
  channels: 62
  num_classes: 3
  topk: 10
  dropout: 0.1

augmentation:
  name: IDPM
  n_steps: 1000
  aug_factor: 1.5
  batch_size: 32

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: Adam
  scheduler: CosineAnnealing
  validation_split: 0.2
```

### Training Scripts

#### 1. Pretrain IDPM

```python
from src.transit_eeg.augmentations.idpm import IDPM
from src.transit_eeg.datasets import SEEDDataset

# Initialize IDPM
idpm = IDPM(
    n_steps=1000,
    channels=62,
    window_size=265,
    device='cuda'
)

# Train IDPM
idpm.train(
    dataset=SEEDDataset(subject_ids=list(range(14))),
    epochs=100,
    batch_size=32
)

# Generate synthetic samples
synthetic_data = idpm.sample(num_samples=100, subject_id=0)
```

#### 2. Train SOGAT

```python
from src.transit_eeg.model.sogat import SOGAT
from torch.utils.data import DataLoader

# Initialize SOGAT
model = SOGAT(
    channels=62,
    num_classes=3,
    dropout=0.1
)

# Train with augmented data
train_loader = DataLoader(augmented_dataset, batch_size=64)
for epoch in range(100):
    for batch in train_loader:
        loss = train_step(model, batch)
```

#### 3. Finetune with LoRA

```python
from src.transit_eeg.model.sogat import SOGAT

# Load pretrained model
model = SOGAT.from_pretrained('checkpoints/pretrained.pt')

# Enable LoRA adapters
model.enable_lora_adapters(rank=8)

# Freeze base parameters
model.freeze_base_layers()

# Finetune on new subject
finetune(model, new_subject_data, epochs=20)
```

## 📈 Results

### SEED Dataset (Emotion Recognition)

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| EEG-GNN | 84.32% | 84.15% | 84.28% | 84.32% |
| EEG-GAT | 85.67% | 85.42% | 85.55% | 85.67% |
| SOGNN | 87.45% | 87.21% | 87.33% | 87.45% |
| **TRANSIT-EEG** | **91.89%** | **91.53%** | **91.71%** | **91.89%** |

### PhyAat Dataset (Auditory Activity)

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| EEG-GNN | 81.23% | 81.05% | 81.14% | 81.23% |
| EEG-GAT | 83.45% | 83.21% | 83.33% | 83.45% |
| SOGNN | 85.67% | 85.42% | 85.55% | 85.67% |
| **TRANSIT-EEG** | **88.12%** | **87.78%** | **87.95%** | **88.12%** |

### Ablation Study

| Component | SEED F1 | PhyAat F1 | Δ |
|-----------|---------|-----------|---|
| SOGAT (Base) | 87.21% | 85.42% | - |
| + IDPM Augmentation | 89.34% | 86.55% | +2.13% |
| + LoRA Finetuning | **91.53%** | **87.78%** | +4.32% |

## 📁 Project Structure

```
transit-eeg/
├── src/
│   └── transit_eeg/
│       ├── __init__.py
│       ├── augmentations/          # IDPM implementation
│       │   ├── idpm.py             # Main IDPM model
│       │   ├── ddpm.py             # Diffusion process
│       │   ├── unet.py             # UNet architecture
│       │   ├── embeddings.py       # Subject embeddings
│       │   └── feature_extractor.py
│       ├── model/                  # SOGAT implementation
│       │   ├── sogat.py            # Main SOGAT model
│       │   ├── sognn.py            # Baseline SOGNN
│       │   └── modules.py          # GAT layers, LoRA adapters
│       ├── datasets/               # Data loaders
│       │   ├── seed_loaders.py
│       │   └── phyaat_loaders.py
│       ├── utils/
│       │   └── utils.py
│       ├── constants.py            # Channel locations
│       └── differential_entropy.py # Feature extraction
├── scripts/                        # Utility scripts
│   ├── download_seed.py
│   ├── download_phyaat.py
│   ├── preprocess.py
│   └── run_loso.py
├── configs/                        # Configuration files
│   ├── seed_pretrain.yaml
│   ├── seed_finetune.yaml
│   ├── phyaat_pretrain.yaml
│   └── phyaat_finetune.yaml
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_idpm_training.ipynb
│   ├── 03_sogat_training.ipynb
│   └── 04_visualization.ipynb
├── train.py                        # Phase 1 training script
├── finetune.py                     # Phase 2 finetuning script
├── evaluate.py                     # Phase 3 evaluation script
├── requirements.txt
└── README.md
```

## 🔬 Methodology Details

### Differential Entropy Feature Extraction

```python
# Extract DE features from 5 frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 49)
}

# For each channel and band
DE = 0.5 * log2(2 * π * e * var(filtered_signal))
```

### Dynamic Graph Construction

```python
# Self-organized adjacency matrix
V = features  # [N_channels, F_features]
W = learnable_weights
G = tanh(V @ W)
A = softmax(G @ G.T)

# Top-k sparsification
A_sparse = topk(A, k=10)
```

### LoRA Decomposition

```python
# Original weight update
ΔW = B @ A  # Full rank

# LoRA decomposition
ΔW ≈ B_low @ A_low  # Low rank
# where B_low: [d, r], A_low: [r, d], r << d
```

## 🎯 Use Cases

1. **Brain-Computer Interfaces (BCI)**: Rapid adaptation to new users
2. **Clinical Applications**: Patient-specific seizure detection
3. **Cognitive Assessment**: Cross-subject emotion recognition
4. **Neurofeedback Systems**: Personalized mental state monitoring

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   batch_size: 16  # Instead of 32
   ```

2. **Import Errors**
   ```bash
   # Reinstall PyTorch Geometric
   pip install --upgrade torch-geometric
   ```

3. **Slow Training**
   ```bash
   # Enable mixed precision training
   python train.py --amp --config configs/seed_pretrain.yaml
   ```

## 📚 Citation

If you use TRANSIT-EEG in your research, please cite:

```bibtex
@article{ahuja2024transit,
  title={TRANSIT-EEG: A Framework for Cross-Subject Classification with Subject Specific Adaptation},
  author={Ahuja, Chirag and Sethia, Divyashikha},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **Chirag Ahuja** - Department of Computer Engineering, Delhi Technology University
- **Divyashikha Sethia** - Department of Software Engineering, Delhi Technology University

## 🙏 Acknowledgments

- SEED dataset providers: SJTU BCMI Lab
- PhyAat dataset: Physionet
- PyTorch Geometric team for graph neural network tools
- The diffusion models community for inspiration

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- Chirag Ahuja: chiragahuja2k20phdco13@dtu.ac.in
- Divyashikha Sethia: divyashikha@dtu.ac.in

---

**Note**: This is research code. For production use, additional testing and optimization may be required.
