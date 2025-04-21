"""
This script performs Leave-One-Subject-Out (LOSO) meta-learning evaluation using the MtlLearner
architecture. It simulates EEG-like data using SimulatedSEED, trains a meta-learning model
on N-1 subjects using episodic few-shot tasks, and evaluates on the held-out subject using
augmented samples as support and query sets.

Main Features:
- Episodic meta-training using FewShotDataset
- Leave-One-Subject-Out training and testing loop
- Support/query generation via augmentation of a single test sample
- Saves model checkpoints, support and query embeddings per fold
- Stores final accuracy summary in JSON
"""

import os
import json
import torch
from transit_eeg.subject_identifier.meta.mtl_learner import MtlLearner
from transit_eeg.subject_identifier.meta.meta_dataloader import FewShotDataset
from transit_eeg.subject_identifier.subject_adapter import SubjectAdapter
from transit_eeg.datasets.seed_sim import SimulatedSEED
from types import SimpleNamespace
import torch.nn.functional as F


print("\n🔁 LOSO Meta-Learning Demo (MtlLearner)")

num_subjects = 6
samples_per_subject = 30
max_test_samples = 1  # use 1 sample per subject and augment for support/query
results_summary = {}

# Step 1: Simulate full dataset with distinguishable embeddings per subject
full_dataset = SimulatedSEED(num_subjects=num_subjects, samples_per_subject=samples_per_subject)

# Step 2: LOSO loop — hold out one subject each round
for leave_out in range(num_subjects):
    print(f"\n🔍 Leaving out subject {leave_out} for testing")

    # Create per-fold output directory
    fold_dir = f"loso_outputs/fold_{leave_out}"
    os.makedirs(fold_dir, exist_ok=True)

    # Split dataset: all but one subject for training, held-out subject for testing
    train_data, test_data = [], []
    for i in range(len(full_dataset)):
        x, y = full_dataset[i]
        (test_data if int(y) == leave_out else train_data).append((x, y))

    # Limit number of held-out samples (only 1 is used and then augmented)
    test_data = test_data[:max_test_samples]

    # Build few-shot meta-learning task dataset from training subjects
    meta_train_ds = FewShotDataset(train_data, way=5, shot=5, query=5, episodes=100)
    meta_train_loader = torch.utils.data.DataLoader(
        meta_train_ds, batch_size=1, shuffle=True
    )

    args = SimpleNamespace(base_lr=0.01, update_step=5, way=5)
    model = MtlLearner(args=args, mode='maml')
    model.meta_train_loop(
        dataloader=meta_train_loader,
        full_dataset=full_dataset,
        epochs=5,
        lr=1e-3,
        save_path=os.path.join(fold_dir, "mtl_checkpoint.pt"),
        embed_path=os.path.join(fold_dir, "all_subject_embeddings.json"),
        device='cpu'
    )

    # Load saved embeddings
    with open(os.path.join(fold_dir, "all_subject_embeddings.json"), "r") as f:
        all_embeddings = json.load(f)
        all_embeddings = {
            int(k.split("_")[1]): torch.tensor(v) for k, v in all_embeddings.items()
        }

    # Few-shot evaluation using augmented single test sample
    print("\n🧪 Few-shot evaluation on held-out subject")
    support_set, query_set = [], []
    held_out_samples = [x for x, y in test_data if int(y) == leave_out]

    if len(held_out_samples) == 0:
        print("⚠️ No samples found for held-out subject.")
        continue

    base_sample = held_out_samples[0]
    support_set = [(base_sample + 0.01 * torch.randn_like(base_sample), 0) for _ in range(5)]
    support_x = torch.stack([x for x, _ in support_set])

    # 🧠 Adapt and extract embedding
    support_subject_embeddings = model.adapt_and_extract_embedding(support_x, inner_steps=5, lr=0.01)
    support_subject_embeddings = F.normalize(support_subject_embeddings, p=2, dim=0)
    all_embeddings[leave_out] = support_subject_embeddings

    base_embeddings = model.embed_query(base_sample)
    base_embeddings = F.normalize(base_embeddings, p=2, dim=0)
    torch.save(support_subject_embeddings, os.path.join(fold_dir, "support_subject_embedding.pt"))

    # 🧪 KNN-style evaluation
    reference_embeddings = torch.stack([emb for emb in all_embeddings.values()])
    scores = F.cosine_similarity(base_embeddings,reference_embeddings, dim=1)
    
    # Identify nearest match
    predicted = torch.argmax(scores).item()
    correct = (predicted == leave_out)
    print(f"🔗 Held-out: {leave_out} → Nearest: {predicted} → {'✅ Correct' if correct else '❌ Wrong'}")

    results_summary[f"subject_{leave_out}"] = {
        "nearest_subject": predicted,
        "correct": correct,
        "accuracy": float(correct)
    }

    print(results_summary)
    print(scores)

    break

# Save overall accuracy summary across all LOSO folds
with open("loso_outputs/summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print("\n📦 Summary saved to loso_outputs/summary.json")
