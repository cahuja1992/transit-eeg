
import torch.nn as nn

class SubjectClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
