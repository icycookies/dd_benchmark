import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer

EMBEDDINGS_PATH = "../data/wordvec/embedding_weibo.npz"

class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_sizes):
        super(CNN1D, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, hidden_size // len(kernel_sizes), (kernel_size, input_size), padding=(2, 0)) for kernel_size in kernel_sizes]
        )

    def forward(self, input):
        h = input.unsqueeze(1)                                      # [batch_size, 1, seq_len, emb_size]
        h = [conv(h).squeeze(3) for conv in self.convs]             # [batch_size, kernel_num, seq_len + 5 - kernel_size] * len(kernel_size)
        h = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in h]      # [batch_size, kernel_num] * len(kernel_size)
        return torch.cat(h, 1)                                      # [batch_size, hidden_size]

class Seq2Feat(nn.Module):
    def __init__(self, max_seq_len, hidden_size, dropout, emb_layer, rnn_layer, num_layers, attention, max_pool=True):
        super(Seq2Feat, self).__init__()
        self.Wudaosb = False
        if emb_layer == "XLNet":
            self.embedding = AutoModel.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
            self.emb_size = 768
        elif emb_layer == "word2vec":
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(EMBEDDINGS_PATH)["embeddings"]))
            self.emb_size = 300
        elif emb_layer == "Wudao":
            self.Wudaosb = True
            self.emb_size = 2560
        self.layer_norm = nn.LayerNorm(normalized_shape=[max_seq_len, self.emb_size])

        if rnn_layer == "CNN1D":
            self.rnn = CNN1D(self.emb_size, hidden_size * 2, [3, 4, 5, 6])
        elif rnn_layer == "BiLSTM":
            self.rnn = nn.LSTM(self.emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        elif rnn_layer == "BiGRU":
            self.rnn = nn.GRU(self.emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        if attention:
            self.att = nn.MultiheadAttention(hidden_size * 2, 8, dropout=dropout)
        else:
            self.att = None

        if max_pool:
            self.pooling = nn.MaxPool1d(max_seq_len)
        else:
            self.pooling = nn.AvgPool1d(max_seq_len)

    def forward(self, seq, mask, user_id=None):
        mask = mask.unsqueeze(2).repeat(1, 1, self.emb_size)  # [batch_size, seq_len, emb_size]
        if self.Wudaosb:
            #print("nmsl")
            h_l = []
            for each in user_id:
                h_l.append(np.load(os.path.join("/home/huangs/depressiondetection/data/weibo2012/processed/WudaoEmbeddings/", str(each.cpu().numpy()) + ".npy")))
            h = np.stack(h_l)
            h = torch.Tensor(h).cuda()
            #print(h.shape)
        else:
            h = self.embedding(seq)[0]
        h = h * mask
        h = self.layer_norm(h)
        if isinstance(self.rnn, CNN1D):
            h = self.rnn(h)
        else:
            h, _ = self.rnn(h)                                  # [batch_size, seq_len, hidden_size*2]
            h = h.transpose(0, 1)                               # [seq_len, batch_size, hidden_size*2]
            if self.att is not None:
                h, _ = self.att(h, h, h)                            
            h = h.permute(1, 2, 0)                              # [batch_size, hidden_size*2, seq_len]
            h = self.pooling(h).squeeze()                       # [batch_size, hidden_size*2]
        return h

class TransformerLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(TransformerLayer, self).__init__()
        self.att = nn.MultiheadAttention(input_size, 8)
        self.gru = nn.GRU(input_size, output_size, dropout=dropout)

    def forward(self, input):
        pass

class TransformerMidLayer(TransformerLayer):
    def __init__(self, input_size, output_size, dropout):
        super(TransformerMidLayer, self).__init__(input_size, output_size, dropout)

    def forward(self, input):
        h = input
        h, _ = self.att(h, h, h)
        h, _ = self.gru(h)                                  # [seq_len, batch_size, output_size]
        return h

class TransformerAggrLayer(TransformerLayer):
    def __init__(self, input_size, output_size, dropout):
        super(TransformerAggrLayer, self).__init__(input_size, output_size, dropout)

    def forward(self, input):
        h = input
        h, _ = self.att(h, h, h)
        h, h_n = self.gru(h)                                # [seq_len, batch_size, output_size]
        return h_n.squeeze(0)                               # [batch_size, output_size]

