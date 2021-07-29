import torch.nn as nn

from models.layers.feature import MLP 

class FeatMLP(nn.Module):
    def __init__(self, num_features, hidden_sizes, dropout):
        super().__init__()
        self.model = MLP(num_features, hidden_sizes, dropout)
        self.classifier = nn.Linear(hidden_sizes[-1], 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, user_avatar_x, microblog_image_x, microblog_image_mask, y, user_id):
        logits = self.classifier(self.model(features_x))
        return logits, self.loss_fn(logits, y)
        #return self.classifier(features_x)