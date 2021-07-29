import torch
import sys
import os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from transformers import AutoModel, AutoTokenizer
from tokenizer import tokenize

def test_xlnet():
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    model = AutoModel.from_pretrained('hfl/chinese-xlnet-base', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    texts = ["你看这个面它又长又宽，就像这个碗它又大又圆。", "我也想过过过儿过过的生活。", "传统功夫讲究化劲啊，点到为止。", "桃李争荣日，荷兰比利时。"]
    texts = [tokenizer.encode(text) for text in texts]
    print([len(text) for text in texts])
    ids = torch.LongTensor([texts[0], texts[1] + [0 for i in range(9)], texts[2] + [0 for i in range(7)], texts[3] + [0 for i in range(10)]])
    print(ids.shape)
    emb = model(ids)
    print(emb[0].shape)
    texts = ["".join("哈" for i in range(512)) for i in range(32)]
    texts = [tokenizer.encode(text) for text in texts]
    ids = torch.LongTensor(texts)
    print(ids.shape)
    emb = model(ids)
    print(emb[0].shape)

def test_word2vec():
    text = "妙蛙种子进米奇妙妙屋，妙到家了"
    print(tokenize(text, "word2vec", 4))

if __name__ == "__main__":
    test_xlnet()
    test_word2vec()
