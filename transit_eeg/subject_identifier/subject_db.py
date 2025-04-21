# TODO: Make a clean subject database with all CRUD operations

import torch
import torch.nn.functional as F
import json
import os

class SubjectDB:
    def __init__(self):
        self.db = {}       # subject_id → list of embeddings
        self.meta = {}     # subject_id → metadata

    def add(self, subject_id, emb, metadata=None):
        if subject_id not in self.db:
            self.db[subject_id] = []
        self.db[subject_id].append(emb.squeeze(0).detach().cpu())
        if metadata:
            self.meta[subject_id] = metadata

    def get_all_embeddings(self):
        return {
            sid: torch.stack(emb_list).mean(dim=0)
            for sid, emb_list in self.db.items()
        }

    def predict(self, emb, metric='cosine', threshold=None):
        emb = emb.squeeze(0).detach().cpu()
        
        best_score = -float('inf')
        best_id = None
        for subject_id, subj_emb in self.get_all_embeddings().items():
            print(subject_id, emb.shape)
            if metric == 'cosine':
                score = F.cosine_similarity(emb, subj_emb, dim=0).item()
            elif metric == 'l2':
                score = -torch.norm(emb - subj_emb).item()
            else:
                raise ValueError("Unknown metric")
            if score > best_score:
                best_score = score
                best_id = subject_id

            print(f"Predicted subject: {subject_id} | Confidence: {score:.2f}")

        if threshold is not None and best_score < threshold:
            return None
        return best_id

    def list_subjects(self, with_meta=False):
        if with_meta:
            return {sid: self.meta.get(sid, {}) for sid in self.db}
        return list(self.db.keys())

    def reset(self):
        self.db.clear()
        self.meta.clear()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.db, os.path.join(path, "embeddings.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(self.meta, f)

    def load(self, path):
        self.db = torch.load(os.path.join(path, "embeddings.pt"))
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
