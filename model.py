import torch.nn as nn
import torch.autograd as autograd
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score


class MF(nn.Module):
    def __init__(self, num_users, num_items, num_feature, num_context, emb_size=100, emb_extra=5, layer_size_1=60,
                 layer_size_2=10, frac=0.2):
        super(MF, self).__init__()
        self.dropout = nn.Dropout(frac)
        self.user_emb = nn.Embedding(num_users, emb_size)
        # self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # self.item_bias = nn.Embedding(num_items, 1)

        self.feature_emb = nn.Embedding(num_feature, emb_extra)
        self.context_emb = nn.Embedding(num_context, emb_extra)

        self.Linear_1 = nn.Linear(emb_size * 2 + emb_extra * 2, layer_size_1)
        self.Linear_2 = nn.Linear(layer_size_1, layer_size_2)
        self.output_ = nn.Linear(layer_size_2, 1)

        # init
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.feature_emb.weight.data.uniform_(0, 0.05)
        self.context_emb.weight.data.uniform_(0, 0.05)
        # self.user_bias.weight.data.uniform_(-0.01,0.01)
        # self.item_bias.weight.data.uniform_(-0.01,0.01)

    def forward(self, u, v, f, c):
        U = self.user_emb(u)
        V = self.item_emb(v)
        F_ = self.feature_emb(f)
        C = self.context_emb(c)
        user_features = torch.cat((U, C), 1)
        item_features = torch.cat((V, F_), 1)
        concat_all = torch.cat((user_features, item_features), 1)
        x = self.Linear_1(concat_all)
        x = F.relu(x)
        x = self.Linear_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_(x)
        return x
