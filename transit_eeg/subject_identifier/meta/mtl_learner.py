import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transit_eeg.subject_identifier.eegnet import EEGNet
import copy

class BaseLearner(nn.Module):
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([args.way, z_dim]))
        nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w, fc1_b = the_vars[0], the_vars[1]
        return F.softmax(F.linear(input_x, fc1_w, fc1_b), dim=1)

    def parameters(self):
        return self.vars


class MtlLearner(nn.Module):
    def __init__(self, args, mode='fomaml', embedding_size=200):
        super().__init__()
        self.args = args
        self.mode = mode.lower() 
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.z_dim = embedding_size
        self.base_learner = BaseLearner(args, self.z_dim)
        self.encoder = EEGNet()

    def forward_meta(self, x_spt, y_spt, x_qry, y_qry):
        task_num, shot_num, *_ = x_spt.size()
        query_num = x_qry.size(1)
        losses_q = []
        corrects = []

        for i in range(task_num):
            emb_spt = self.encoder(x_spt[i])
            emb_qry = self.encoder(x_qry[i])
            fast_weights = list(self.base_learner.parameters())
            grads_list = []

            for k in range(self.update_step):
                logits = self.base_learner(emb_spt, fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                create_graph = (self.mode == 'maml')
                grads = torch.autograd.grad(loss, fast_weights, create_graph=create_graph)
                fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grads)]
                if self.mode == 'reptile':
                    grads_list.append([g.clone().detach() for g in fast_weights])

            logits_q = self.base_learner(emb_qry, fast_weights)
            loss_q = F.cross_entropy(logits_q, y_qry[i])
            pred_q = logits_q.argmax(dim=1)
            acc_q = torch.eq(pred_q, y_qry[i]).sum().item() / y_qry[i].size(0)

            losses_q.append(loss_q)
            corrects.append(acc_q)

            # REPTILE update (just average the parameters)
            if self.mode == 'reptile':
                reptile_step = grads_list[-1]
                with torch.no_grad():
                    for w, w_tgt in zip(self.base_learner.vars, reptile_step):
                        w.data = w.data + self.update_lr * (w_tgt.data - w.data)

        return torch.stack(losses_q).mean(), sum(corrects) / len(corrects)
    

    def meta_train_loop(self, dataloader, full_dataset, epochs=5, lr=1e-3, save_path=None, embed_path=None, device='cuda'):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            losses, accs = [], []
            for batch in dataloader:
                x_spt, y_spt, x_qry, y_qry = [t.squeeze(0).to(device) for t in batch]
                loss, acc = self.forward_meta(x_spt, y_spt, x_qry, y_qry)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                accs.append(acc)
            print(f"Epoch {epoch+1}: Loss={sum(losses)/len(losses):.4f} | Acc={sum(accs)/len(accs):.2%}")

        if save_path is not None:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)

        if embed_path is not None:
            self.adapted_encoder = self.encoder
            all_embeddings = {}
            for subject_id in range(6):
                subject_data = [x for x, y in full_dataset if int(y) == subject_id]
                if not subject_data:
                    continue
                support_x = torch.stack(subject_data)
                subject_embedding = self.embed_query(support_x).mean(dim=0) # centroids per subject
                all_embeddings[f"subject_{subject_id}"] = subject_embedding.tolist()

            with open(embed_path, "w") as f:
                json.dump(all_embeddings, f, indent=2)
            print(f"\n📦 All subject embeddings saved to {embed_path}")

    
    def adapt_and_extract_embedding(self, x_spt, y_spt=None, inner_steps=5, lr=0.01):
        x_spt = x_spt.to(self.encoder.device if hasattr(self.encoder, 'device') else 'cpu')
        adapted_encoder = copy.deepcopy(self.encoder).to(x_spt.device)
        adapted_encoder.train()
        optimizer = torch.optim.SGD(adapted_encoder.parameters(), lr=lr)

        for _ in range(inner_steps):
            out = adapted_encoder(x_spt)
            if out.dim() > 2:
                out = out.flatten(start_dim=1)
            if y_spt is None:
                y_spt = torch.zeros(len(x_spt), dtype=torch.long, device=x_spt.device)
            loss = F.cross_entropy(out, y_spt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.adapted_encoder = adapted_encoder
        adapted_encoder.eval()
        with torch.no_grad():
            emb = adapted_encoder(x_spt)
            if emb.dim() > 2:
                emb = emb.flatten(start_dim=1)
            return emb.mean(dim=0).cpu()

    def embed_query(self, x_qry,):
        if x_qry.dim() == 3:
            x_qry = x_qry.unsqueeze(0)  # ensure batch dimension
        x_qry = x_qry.to(self.adapted_encoder.device if hasattr(self.adapted_encoder, 'device') else 'cpu')
        with torch.no_grad():
            out = self.adapted_encoder(x_qry)
            if out.dim() > 2:
                out = out.flatten(start_dim=1)
            return out.cpu()

