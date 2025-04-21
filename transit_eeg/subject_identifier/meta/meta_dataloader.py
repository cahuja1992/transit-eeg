# transit_eeg/subject_identifier/meta/meta_dataloader.py

import torch
import numpy as np
from torch.utils.data import Dataset
import random

class FewShotDataset(Dataset):
    def __init__(self, dataset, way=5, shot=5, query=15, episodes=1000):
        self.dataset = dataset
        self.way = way
        self.shot = shot
        self.query = query
        self.episodes = episodes
        self.class_to_indices = self.build_class_index()

    def build_class_index(self):
        label_map = {}
        for i in range(len(self.dataset)):
            _, y = self.dataset[i]
            y = int(y)
            if y not in label_map:
                label_map[y] = []
            label_map[y].append(i)
        return label_map

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        selected_classes = random.sample(list(self.class_to_indices.keys()), self.way)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for label_idx, cls in enumerate(selected_classes):
            indices = random.sample(self.class_to_indices[cls], self.shot + self.query)
            support_idx = indices[:self.shot]
            query_idx = indices[self.shot:]

            support_x.extend([self.dataset[i][0].unsqueeze(0) for i in support_idx])
            support_y.extend([label_idx] * self.shot)
            query_x.extend([self.dataset[i][0].unsqueeze(0) for i in query_idx])
            query_y.extend([label_idx] * self.query)

        support_x = torch.cat(support_x, dim=0).view(self.way, self.shot, *support_x[0].shape[1:])
        query_x = torch.cat(query_x, dim=0).view(self.way, self.query, *query_x[0].shape[1:])
        support_y = torch.tensor(support_y).view(self.way, self.shot)
        query_y = torch.tensor(query_y).view(self.way, self.query)
        return support_x, support_y, query_x, query_y
