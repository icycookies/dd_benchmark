from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from transformers import AutoModel

from models.layers.feature import MLP
from models.layers.image import ResNet18
from models.layers.text import Seq2Feat, TransformerMidLayer, TransformerAggrLayer
from utils import make_input

logger = logging.getLogger(__name__)

def pretrain_PMFN(args, train_dataset, model):
    sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.pretrain_batch_size)
    optimizer = torch.optim.Adam(model.recon_mlp.parameters(), lr=args.lr)
    iterator = tqdm(range(args.pretrain_epochs), desc="Epoch")
    model.pretraining = True

    logger.info("**** Running PreTraining ****")
    logger.info("   Num epochs = %d", args.pretrain_epochs)
    logger.info("   Num samples = %d", len(train_dataset))
    logger.info("   Batch size = %d", args.pretrain_batch_size)
    logger.info("   Num steps = %d", args.pretrain_epochs * len(train_dataset) // args.pretrain_batch_size)
    
    global_step = 0
    average_loss = 0
    for _ in iterator:
        for batch in dataloader:
            global_step += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = make_input(batch)
            loss = model(**inputs)
            average_loss += loss.item()
            if args.pretrain_gradient_accumulation_steps > 1:
                loss /= args.pretrain_gradient_accumulation_steps
            loss.backward()

            if global_step % args.pretrain_gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

            if global_step % args.logging_steps == 0:
                logger.info("Average loss %s at global step %s", str(average_loss / args.logging_steps), str(global_step)) 
                average_loss = 0

    model.pretraining = False    
    return model

# Pretrained Multi-Modal Fusion Network for Depression Detection
class PMFN(nn.Module):
    def __init__(self, num_features, hidden_sizes, dropout, text_emb_layer, visual_emb_layer, max_seq_len, max_microblogs, num_aggr_layers, w1, w2):
        super().__init__()
        self.text_emb_size = 768
        self.hidden_size = hidden_sizes["hidden_size"]

        self.embedding = AutoModel.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
        self.layer_norm = nn.LayerNorm(normalized_shape=[1024, self.text_emb_size])
        if visual_emb_layer == "ResNet18":
            self.visual_embedding = ResNet18(dropout)
            self.visual_emb_size = 512

        self.recon_mlp = MLP(self.text_emb_size, hidden_sizes["recon"], dropout)
        transformer_sizes = [1280] + [hidden_sizes["hidden_size"] for i in range(num_aggr_layers)]
        self.microblog2feat = nn.ModuleList(
            [TransformerMidLayer(transformer_sizes[i], transformer_sizes[i + 1], dropout) for i in range(num_aggr_layers - 1)] + 
            [TransformerAggrLayer(transformer_sizes[-2], transformer_sizes[-1], dropout)]
        )

        self.batch_norm = nn.BatchNorm1d(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + num_features + self.text_emb_size + self.visual_emb_size, hidden_sizes["classifier"][0]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes["classifier"][0], 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.max_microblogs = max_microblogs
        self.max_seq_len = max_seq_len
        self.w1 = w1
        self.w2 = w2
        self.pretraining = False

    def forward(self, features_x, user_text_x, user_text_att_mask, microblog_text_x, microblog_text_att_mask, user_avatar_x, microblog_image_x, microblog_image_mask, y, user_id):
        """
        Input Tensor Size:
        features_x:              [batch_size, num_features]
        user_text_x:             [batch_size, seq_len]
        user_text_att_mask:      [batch_size, seq_len]
        microblog_text_x:        [batch_size, max_microblogs, seq_len]
        microblog_text_att_mask: [batch_size, max_microblogs, seq_len]
        user_avatar_x:           [batch_size, channels, H, W]
        microblog_image_x:       [batch_size, max_microblogs, channels, H, W]
        microblog_image_mask:    [batch_size, max_microblogs]
        y:                       [batch_size]
        """
        h_text = self.embedding(user_text_x)[0]
        # h_text = self.layer_norm(h_text)
        # [batch_size, seq_len, text_emb_size]
        h_user_text = h_text[:, 1, :]
        # [batch_size, text_emb_size]
        h_text = torch.matmul(user_text_att_mask, h_text)[:, 1:, :].reshape((-1, self.text_emb_size))
        # [batch_size * max_microblogs, text_emb_size]
        h_recon = self.recon_mlp(h_text)
        # [batch_size * max_microblogs, visual_emb_size]

        h_image = self.visual_embedding(microblog_image_x.view(-1, 3, 224, 224))
        # [batch_size * max_microblogs, visual_emb_size]
        indexs = torch.where(microblog_image_mask.view(-1) == 1)
        
        #recon_loss = self.recon_loss(h_recon[indexs], h_image[indexs])

        if self.pretraining:
            return self.recon_loss(h_recon[indexs], h_image[indexs])

        microblog_image_mask = microblog_image_mask.view(-1).unsqueeze(1).repeat(1, self.hidden_size)
        # [batch_size * max_microblogs, visual_emb_size]
        h_image = h_image * microblog_image_mask + h_recon * (1 - microblog_image_mask)

        h = torch.cat((h_text, h_image), dim=1)
        h = h.view(-1, self.max_microblogs, self.text_emb_size + self.visual_emb_size).transpose(0, 1)
        # [max_microblogs, batch_size, text_emb_size + visual_emb_size]
        for layer in self.microblog2feat:
            h = layer(h)
        # [batch_size, hidden_size]

        h_image = self.visual_embedding(user_avatar_x)
        features_x = self.batch_norm(features_x)
        h = torch.cat((h, h_user_text, h_image, features_x), 1)
        # [batch_size, hidden_size + text_emb_size + visual_emb_size + num_features]
        output = self.classifier(h)
        classification_loss = self.loss_fn(output, y)
        return output, classification_loss
        #return output, self.w1 * recon_loss + self.w2 * classification_loss

    def recon_loss(self, text_feat, visual_feat):
        return nn.MSELoss(reduction="sum")(text_feat, visual_feat)
    