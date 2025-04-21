# subject_adapter.py
import torch
import torch.nn.functional as F
import copy

class SubjectAdapter:
    """
    Wrapper to adapt a meta-learned model like MtlLearner or EEGNet to a new subject
    and extract subject embeddings for downstream use.
    """
    def __init__(self, model, embedding_fn=None, device='cuda'):
        """
        Args:
            model: A model with encoder and forward methods (e.g., MtlLearner, EEGNet)
            embedding_fn: Optional callable to extract embeddings from model output
            device: Torch device
        """
        self.model = model.to(device)
        self.device = device
        self.embedding_fn = embedding_fn if embedding_fn is not None else self.default_embedding_fn

    def default_embedding_fn(self, x):
        """Extract embeddings using the encoder and flatten if needed."""
        with torch.no_grad():
            out = self.model.encoder(x.to(self.device))
            if out.dim() > 2:
                out = torch.flatten(out, start_dim=1)
            return out.cpu()

    def adapt(self, x_spt, y_spt, inner_steps=5, lr=0.01):
        """
        Finetune a copy of the model on a small support set.

        Args:
            x_spt: Support inputs (B, C, H, W)
            y_spt: Support labels (B,)
            inner_steps: Number of gradient steps
            lr: Inner loop learning rate

        Returns:
            Adapted encoder model
        """
        adapted_encoder = copy.deepcopy(self.model.encoder).to(self.device)
        optimizer = torch.optim.SGD(adapted_encoder.parameters(), lr=lr)
        
        x_spt = x_spt.to(self.device)
        y_spt = y_spt.to(self.device)

        for _ in range(inner_steps):
            logits = adapted_encoder(x_spt)
            if logits.dim() > 2:
                logits = torch.flatten(logits, start_dim=1)
            loss = F.cross_entropy(logits, y_spt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        return adapted_encoder

    def extract_subject_embedding(self, x_spt, adapt=True, **adapt_kwargs):
        """
        Extract an embedding for a new subject.

        Args:
            x_spt: Support set inputs for the subject
            adapt: If True, adapt model to the support set first
            adapt_kwargs: Keyword arguments for the `adapt` method

        Returns:
            A subject-level embedding tensor
        """
        if adapt:
            encoder = self.adapt(x_spt, torch.zeros(len(x_spt), dtype=torch.long), **adapt_kwargs)
        else:
            encoder = self.model.encoder

        encoder.eval()
        with torch.no_grad():
            features = encoder(x_spt.to(self.device))
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            return features.mean(dim=0).cpu()

    def save_embedding(self, embedding, path):
        torch.save(embedding, path)

    def load_embedding(self, path):
        return torch.load(path)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using the adapted model."""
        self.adapted_model.eval()
        with torch.no_grad():
            logits = self.adapted_model(x)
            return torch.argmax(logits, dim=-1)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Return classification accuracy."""
        preds = self.predict(x)
        return (preds == y).float().mean().item()
