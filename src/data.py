import json
import logging
import os
import os.path as osp
from functools import partial

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from features import generate_predefined_features
from tokenizer import tokenize
from utils import (convert_image_to_tensor, download_picture, filter_text,
                   str_to_time, generate_masks)


logger = logging.getLogger(__name__)
supported_datasets = {"weibo2012", "wu3d"}
open = partial(open, encoding='utf8')   # 方便在 windows 上测试

POSITIVE, NEGATIVE = 1, 0


class User(object):
    def __init__(self, user, type, download_picture):
        self.id = user['idstr']
        self.name = user['name']
        self.gender = 0 if user['gender'] == 'f' else 1
        self.description = user['description'] if 'description' in user else "",
        self.followers_count = user['followers_count']
        self.friends_count = user['friends_count']
        self.bi_followers_count = user['bi_followers_count']
        self.avatar_url = user['avatar_large']
        self.avatar_path = osp.join("../data/weibo2012/pic/avatar", self.id) + ".jpg"
        self.type = type
        self.emoticon_count = [0, 0, 0]

        if self.avatar_url is not None and not osp.exists(self.avatar_path) and download_picture:
            download_picture(self.avatar_url, self.avatar_path)


class Microblog(object):
    def __init__(self, microblog_data, cache_dir, download_picture):
        self.id = microblog_data['idstr']
        self.text = microblog_data['text']
        self.retweet = 'retweeted_status' in microblog_data
        self.image_url = microblog_data['bmiddle_pic'] if 'bmiddle_pic' in microblog_data else None
        self.image_path = osp.join('../data/weibo2012/pic/microblogs', self.id) + ".jpg" if self.image_url is not None else None
        self.time_str = microblog_data['created_at']

        if self.image_path is not None and not osp.exists(self.image_path) and download_picture:
            download_picture(self.image_url, self.image_path)
        self.year, self.month, self.day, self.day_of_the_week, self.hour, self.minute, self.minute = str_to_time(self.time_str)

        self.text, self.emoticon_count = filter_text(self.text)


class WeiboDataset(Dataset):
    def __init__(
        self,
        name="weibo2012",
        download_picture=False,
        overwrite_cache=True,
        tokenizer="word2vec",
        user_only=False,
        max_seq_len=64,
        max_microblogs=128,
        process_image=False
    ):
        if name not in supported_datasets:
            logger.error("{} dataset not exists!".format(name))
            raise NotImplementedError
        self.name = name
        data_dir = osp.join("../data", name, "raw")
        cache_dir = osp.join("../data", name, "processed")

        self.user_id = []
        self.microblog_id = []

        self.features_x = []
        self.user_text_x = []
        self.user_text_att_mask = []
        self.microblog_text_x = []
        self.microblog_text_att_mask = []
        self.user_avatar_x = []
        self.y = []
        self.microblog_start = []
        self.user_only = user_only
        self.process_image = process_image
        self.max_microblogs = max_microblogs
        self.max_seq_len = max_seq_len

        self.cache_path = osp.join(cache_dir, f"{name}_data_{tokenizer}_{max_seq_len}_{max_microblogs}_{'user' if self.user_only else 'microblog'}")
        if not osp.exists(self.cache_path) or overwrite_cache:
            self.users = []
            self.microblogs = {}
            with open(osp.join(data_dir, 'positive.txt'), "r") as f:
                self._read_raw_data(f, POSITIVE, cache_dir, download_picture)
            with open(osp.join(data_dir, 'negative.txt'), "r") as f:
                self._read_raw_data(f, NEGATIVE, cache_dir, download_picture)
            self._process_data(cache_dir)
            self._tokenize(cache_dir, tokenizer)
            self._save_to_cache(cache_dir, tokenizer)
        else:
            self._read_from_cache(cache_dir, tokenizer)

        self._process_images(cache_dir, process_image)

    def _read_raw_data(self, fin, user_type, cache_dir, download_picture):
        # read user and microblog info from raw json file
        lines = fin.readlines()

        for _, data_raw in tqdm(enumerate(lines), desc="reading raw data and downloading images"):
            data = json.loads(data_raw)
            # user information
            user = data['user']
            user_id = user['idstr']
            microblog_id = []
            self.users.append(User(user, user_type, download_picture))

            self.microblogs[user_id] = []
            for microblog_data in data['microblogs']:
                self.microblogs[user_id].append(Microblog(microblog_data, cache_dir, download_picture))
                for i in range(3):
                    self.users[-1].emoticon_count[i] += self.microblogs[user_id][-1].emoticon_count[i]
                microblog_id.append(int(self.microblogs[user_id][-1].id))

            self.microblogs[user_id].reverse()
            if len(self.microblogs[user_id]) > self.max_microblogs:
                self.microblogs[user_id] = self.microblogs[user_id][:self.max_microblogs]
                microblog_id = microblog_id[:self.max_microblogs]
            else:
                microblog_id = microblog_id + [0 for i in range(self.max_microblogs - len(microblog_id))]
            self.microblog_id.append(torch.LongTensor(microblog_id))

            cache_text_dir = osp.join(cache_dir, "texts")
            if not osp.exists(cache_text_dir):
                os.mkdir(cache_text_dir)
            with open(osp.join(cache_text_dir, user_id + ".txt"), "w") as f:
                s = filter_text("{}{}".format(user['name'], user['description']))[0] + "\n" + "\n".join([filter_text(microblog.text)[0] for microblog in self.microblogs[user_id]])
                f.write(s)

    def _process_data(self, cache_dir):
        # generate features, labels for users
        for idx, user in tqdm(enumerate(self.users), desc="processing data"):
            self.user_id.append(int(user.id))

            self.features_x.append(generate_predefined_features(user, self.microblogs[user.id], cache_dir))
            self.y.append(user.type)

            if idx < 2:
                logger.info("*** Example User #{} ***".format(idx))
                logger.info("user_id: {}".format(user.id))
                logger.info("name: {}".format(user.name))
                logger.info("gender: {}".format("female" if user.gender == 0 else "male"))
                logger.info("desciption: {}".format(user.description))
                logger.info("followers_count: {}".format(user.followers_count))
                logger.info("bi_followers_count: {}".format(user.bi_followers_count))
                logger.info("friends_count: {}".format(user.friends_count))
                logger.info("num_microblogs: {}".format(len(self.microblogs[user.id])))
                logger.info("--- Example microblogs ---")
                for i in range(2):
                    microblog = self.microblogs[user.id][i]
                    logger.info("time: {}".format(microblog.time_str))
                    logger.info("text: {}".format(microblog.text))
                    logger.info("is_retweet: {}".format(microblog.retweet))
                logger.info("predefined_features: {}".format(self.features_x[-1]))
                logger.info("label: {}".format(self.y[-1]))

        self.user_id = torch.LongTensor(self.user_id)
        self.microblog_id = torch.stack(self.microblog_id)
        self.features_x = torch.stack(self.features_x)
        self.y = torch.LongTensor(self.y)

    def _tokenize(self, cache_dir, tokenizer):
        for _, id in tqdm(enumerate(self.user_id.numpy()), desc="tokenizing"):
            with open(osp.join(cache_dir, "texts", str(id) + ".txt"), "r") as f:
                if self.user_only:
                    texts = "".join(f.readlines())
                    tokens, att = tokenize(texts, tokenizer, self.max_seq_len)
                    self.user_text_x.append(tokens)
                    self.user_text_att_mask.append(att)
                else:
                    tokens_all, _ = tokenize(f.readline(), tokenizer, self.max_seq_len, False)
                    microblog_start = [0, tokens_all.shape[0]]

                    microblog_tokens, microblog_att = [], []
                    for text in f.readlines():
                        tokens, att = tokenize(text, tokenizer, self.max_seq_len)
                        microblog_tokens.append(tokens)
                        microblog_att.append(att)

                        if len(tokens_all) < 1024:
                            tokens_all = torch.cat((tokens_all, tokenize(text, tokenizer, min(self.max_seq_len, 1024 - len(tokens_all)), False)[0]))
                            microblog_start.append(len(tokens_all))

                    if len(microblog_tokens) < self.max_microblogs:
                        microblog_tokens += [torch.zeros(self.max_seq_len, dtype=torch.int64) for i in range(self.max_microblogs - len(microblog_tokens))]
                        microblog_att += [torch.zeros(self.max_seq_len) for i in range(self.max_microblogs - len(microblog_att))]
                    assert len(microblog_tokens) == self.max_microblogs
                    assert len(microblog_att) == self.max_microblogs

                    if len(microblog_start) < self.max_microblogs + 2:
                        val = microblog_start[-1]
                        microblog_start += [val for i in range(self.max_microblogs + 2 - len(microblog_start))]
                    if len(tokens_all) < 1024:
                        tokens_all = torch.cat((tokens_all, torch.LongTensor([0 for i in range(1024 - len(tokens_all))])))
                        att = [1.0 for i in range(len(tokens_all))] + [0.0 for i in range(1024 - len(tokens_all))]
                    else:
                        att = [1.0 for i in range(1024)]
                    
                    # print(len(tokens_all), len(att), microblog_start)

                    self.microblog_text_x.append(torch.stack(microblog_tokens))
                    self.microblog_text_att_mask.append(torch.stack(microblog_att))

                    self.user_text_x.append(tokens_all)
                    self.user_text_att_mask.append(torch.FloatTensor(att))
                    self.microblog_start.append(torch.LongTensor(microblog_start))

        self.user_text_x = torch.stack(self.user_text_x)
        self.user_text_att_mask = torch.stack(self.user_text_att_mask)
        logger.info("user text shape: [{}]".format(", ".join(map(str, self.user_text_x.shape))))

        if not self.user_only:
            self.microblog_text_x = torch.stack(self.microblog_text_x)
            self.microblog_text_att_mask = torch.stack(self.microblog_text_att_mask)
            self.microblog_start = torch.stack(self.microblog_start)
            logger.info("microblog text shape: {} [{}]".format(str(self.microblog_text_x.dtype), ", ".join(map(str, self.microblog_text_x.shape))))
        else:
            length = len(self)
            self.microblog_text_x = torch.zeros(length)
            self.microblog_text_att_mask = torch.zeros(length)
            self.microblog_start = torch.zeros(length)

    def _process_images(self, cache_dir, process_image):
        if process_image:
            logger.info("processing images")
            for id in self.user_id.numpy():
                avatar_path = osp.join("../data", self.name, "pic/avatar", str(id)) + ".jpg"
                self.user_avatar_x.append(convert_image_to_tensor(avatar_path))
            self.user_avatar_x = torch.stack(self.user_avatar_x)
            logger.info("user avatar shape: [{}]".format(", ".join(map(str, self.user_avatar_x.shape))))
        else:
            self.user_avatar_x = torch.zeros(len(self))
        """
        do not process microblog images because padding is too large
        2466 * 128 * 3 * 224 * 224
        """

    def _save_to_cache(self, cache_dir, tokenizer):
        processed = {'user_id': self.user_id,
                     'microblog_id': self.microblog_id,
                     'features_x': self.features_x,
                     'user_text_x': self.user_text_x,
                     'user_text_att_mask': self.user_text_att_mask,
                     'microblog_text_x': self.microblog_text_x,
                     'microblog_text_att_mask': self.microblog_text_att_mask,
                     'y': self.y,
                     'microblog_start': self.microblog_start
                     }
        torch.save(processed, self.cache_path)

    def _read_from_cache(self, cache_dir, tokenizer):
        logger.info("Loading {} dataset from cache".format(self.name))
        cached_data = torch.load(self.cache_path)
        self.user_id = cached_data['user_id']
        self.microblog_id = cached_data['microblog_id']
        self.features_x = cached_data['features_x']
        self.user_text_x = cached_data['user_text_x']
        self.user_text_att_mask = cached_data['user_text_att_mask']
        self.microblog_text_x = cached_data['microblog_text_x']
        self.microblog_text_att_mask = cached_data['microblog_text_att_mask']
        self.y = cached_data['y']
        self.microblog_start = cached_data['microblog_start']

    def __getitem__(self, index):
        if self.process_image:
            microblog_image_x = []
            microblog_image_mask = []
            for id in self.microblog_id[index].numpy():
                path = osp.join("../data", self.name, "pic/microblogs", str(id)) + ".jpg"
                if osp.exists(path):
                    microblog_image_x.append(convert_image_to_tensor(path))
                    microblog_image_mask.append(1)
                else:
                    microblog_image_x.append(torch.zeros(3, 224, 224))
                    microblog_image_mask.append(0)
            microblog_image_x = torch.stack(microblog_image_x)
            microblog_image_mask = torch.FloatTensor(microblog_image_mask)
        else:
            microblog_image_x = torch.zeros(1)
            microblog_image_mask = torch.zeros(1)

        if len(self.microblog_start.shape) > 1:
            user_text_att_mask = generate_masks(self.microblog_start[index], 1024)
        else:
            user_text_att_mask = self.user_text_att_mask[index]

        return self.features_x[index], self.user_text_x[index], user_text_att_mask, self.microblog_text_x[index], self.microblog_text_att_mask[index], self.user_avatar_x[index], microblog_image_x, microblog_image_mask, self.y[index], self.user_id[index]

    def __len__(self):
        return self.y.size(0)


def test_wu3d():
    logger.info = print
    dataset = WeiboDataset(name="wu3d")
    print(len(dataset), dataset[1])


if __name__ == '__main__':
    test_wu3d()
