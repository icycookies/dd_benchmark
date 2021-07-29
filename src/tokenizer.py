import os
import pickle

import numpy as np
import pkuseg
import torch
from transformers import AutoTokenizer

WORD2VEC_PATH = "../data/wordvec/sgns.weibo.bigram-char"
EMBEDDINGS_PATH = "../data/wordvec/embedding_weibo.npz"
VOCAB_PATH = "../data/wordvec/vocab.pkl"

UNK, PAD = '[UNK]', '[PAD]'
PAD_TOKEN = 0

WORD2ID = None
seg = pkuseg.pkuseg(model_name='web')

XLNetTokenizer = AutoTokenizer.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')


def word2vec(text, length_limit):
    global seg
    words = seg.cut(text)[:length_limit]
    res = [WORD2ID.get(word, 1) for word in words]
    return res


def init_embeddings():
    global WORD2ID

    if os.path.exists(VOCAB_PATH) and os.path.exists(EMBEDDINGS_PATH):
        WORD2ID = pickle.load(open(VOCAB_PATH, 'rb'))
    else:
        assert os.path.exists(WORD2VEC_PATH), f"word2vec file not exists in {os.getcwd()}"
        WORD2ID, embeddings = {UNK: 1, PAD: 0}, None
        with open(WORD2VEC_PATH, 'r', encoding='utf8') as f:
            words_cnt, dim = f.readline().strip().split()
            words_cnt, dim = int(words_cnt), int(dim)
            embeddings = np.zeros((words_cnt + 2, dim))

            for idx, line in enumerate(f, 2):
                words = line.strip().split(' ')
                WORD2ID[words[0]] = idx
                emb = [float(x) for x in words[1:]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump(WORD2ID, f)
        np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings)


def tokenize(text, tokenizer, length_limit=128, padding=True):
    if length_limit == 2:
        return torch.LongTensor([4, 3]), torch.FloatTensor([1, 1])
    elif length_limit == 1:
        return torch.LongTensor([3]), torch.FloatTensor([1])
    if tokenizer == "word2vec":
        input_ids = word2vec(text, length_limit)
    elif tokenizer == "xlnet":
        input_ids = XLNetTokenizer.encode(text)
    elif tokenizer == "Wudao":
        print("cnm tokenizer Wudaosb")

    if len(input_ids) > length_limit:
        input_ids = input_ids[:length_limit - 2] + input_ids[-2:]
        attention_mask = np.ones(length_limit)
    elif padding:
        attention_mask = [1.0 for i in range(len(input_ids))] + [0.0 for i in range(length_limit - len(input_ids))]
        input_ids = input_ids + [PAD_TOKEN for i in range(length_limit - len(input_ids))]

        assert len(input_ids) == length_limit
        assert len(attention_mask) == length_limit
    else:    
        attention_mask = np.ones(len(input_ids))
    return torch.LongTensor(input_ids), torch.FloatTensor(attention_mask)


init_embeddings()
