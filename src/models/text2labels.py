import torch
import torch.nn as nn

from models.layers.text import CNN1D, Seq2Feat, TransformerMidLayer, TransformerAggrLayer
from transformers import AutoModel

class UserText2Labels(nn.Module):
    def __init__(self, dropout, hidden_size, max_seq_len, emb_layer, proc_layer, num_layers, attention):
        super(UserText2Labels, self).__init__()   
        self.seq2feat = Seq2Feat(max_seq_len, hidden_size, dropout, emb_layer, proc_layer, num_layers, attention) 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, user_avatar_x, microblog_image_x, microblog_image_mask, y, user_id):
        h = self.seq2feat(user_text_x, user_text_att_mask)
        logits = self.classifier(h)
        return logits, self.loss_fn(logits, y)

class MicroblogText2Labels(nn.Module):
    def __init__(self, dropout, hidden_sizes, batch_size, max_seq_len, max_microblogs, emb_layer, proc_layer, num_emb_layers, num_aggr_layers):
        super(MicroblogText2Labels, self).__init__()
        self.emb_size = 768
        self.max_seq_len = max_seq_len
        self.max_microblogs = max_microblogs

        self.embedding = AutoModel.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
        self.layer_norm = nn.LayerNorm(normalized_shape=[1024, self.emb_size])
        transformer_sizes = [768] +[hidden_sizes["text"] for i in range(num_aggr_layers)] 
        self.aggr = nn.ModuleList(
            [TransformerMidLayer(transformer_sizes[i], transformer_sizes[i + 1], dropout) for i in range(num_aggr_layers - 1)] +
            [TransformerAggrLayer(transformer_sizes[-2], transformer_sizes[-1], dropout)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes["text"], hidden_sizes["classifier"]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_sizes["classifier"], 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, user_avatar_x, microblog_image_x, microblog_image_mask, y, user_id):
        """
        Used Tensor Size:
        user_text_x:             [batch_size, seq_len]
        user_text_att_mask:      [batch_size, max_microblogs, seq_len]
        microblog_text_x:        [batch_size, max_microblogs, seq_len]
        microblog_text_att_mask: [batch_size, max_microblogs, seq_len]
        y:                       [batch_size]
        """
        h = self.embedding(user_text_x)[0]
        h = self.layer_norm(h)
        # [batch_size, seq_len, emb_size]
        h = torch.matmul(user_text_att_mask, h).transpose(0, 1)
        # [max_microblogs + 1, batch_size, emb_size]

        for layer in self.aggr:
            h = layer(h)
        # [batch_size, hidden_size]
        logits = self.classifier(h)
        return logits, self.loss_fn(logits, y)