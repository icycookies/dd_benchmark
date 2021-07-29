import torch
import torch.nn as nn

from models.layers.feature import MLP
from models.layers.text import Seq2Feat

class FusionNet(nn.Module):
    def __init__(self, num_features, hidden_sizes, dropout, emb_layer, mid_layer, attention, max_seq_len, w1, w2):
        super(FusionNet, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.seq2feat = Seq2Feat(max_seq_len, hidden_sizes["text"], dropout, emb_layer, mid_layer, 1, attention)
        self.linear = nn.Linear(hidden_sizes["text"] * 2, hidden_sizes["task1"])
        self.mlp = MLP(hidden_sizes["text"] * 2 + num_features, hidden_sizes["task2"], dropout)
        self.classifier1 = nn.Linear(hidden_sizes["task1"], 2)
        self.classifier2 = nn.Linear(hidden_sizes["task2"][-1], 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, user_id, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, 
                user_avatar_x, microblog_image_x, microblog_image_mask, y):
        h = self.seq2feat(user_text_x, user_text_att_mask, user_id)
        out1 = self.linear(h)

        feat_h = self.batch_norm(features_x)
        h = torch.cat((h, feat_h), 1)                           # [batch_size, hidden_size*2 + num_feats]
        out2 = self.mlp(h)

        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)
        return out2, self.w1 * self.loss_fn(out1, y) + self.w2 * self.loss_fn(out2, y)
