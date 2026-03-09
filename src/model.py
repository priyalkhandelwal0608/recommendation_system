import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_fc = nn.Linear(embedding_dim, embedding_dim)
        self.item_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        user_vec = self.user_fc(user_vec)
        item_vec = self.item_fc(item_vec)
        return user_vec, item_vec