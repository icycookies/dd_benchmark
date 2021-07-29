import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.image import ResNet18

class ResNet(nn.Module):
    def __init__(self, dropout, max_microblogs):
        super().__init__()
        self.hidden_size = 512
        self.max_microblogs = max_microblogs

        self.model = ResNet18(dropout)
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.pool = nn.AvgPool1d(max_microblogs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, user_avatar_x, microblog_image_x, microblog_image_mask, y, user_id):
        h_user_x = self.model(user_avatar_x)
        # [batch_size, hidden_size]
        h_microblog_x = self.model(microblog_image_x.view(-1, 3, 224, 224)).view(-1, self.max_microblogs, self.hidden_size)
        # [batch_size, max_microblogs, hidden_size]
        h_microblog_x = (h_microblog_x * microblog_image_mask.unsqueeze(2).repeat(1, 1, self.hidden_size)).transpose(1, 2)
        # [batch_size, hidden_size, max_microblogs]
        bias = (self.max_microblogs / (microblog_image_mask.sum(dim=1) + 1)).unsqueeze(1).repeat(1, self.hidden_size)
        # [batch_size, hidden_size]
        h_microblog_x = self.pool(h_microblog_x).squeeze()
        h_microblog_x = h_microblog_x * bias
        # [batch_size, hidden_size]
        logits = self.classifier(torch.cat((h_user_x, h_microblog_x), 1))
        return logits, self.loss_fn(logits, y)