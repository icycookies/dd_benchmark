import os
import re
import time

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms


str_to_month = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,
                "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
str_to_day_of_the_week = {"Mon": 1, "Tue": 2,
                          "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
emojis = {
    "pos_emoji": "(\[哈哈\]|\[偷笑\]|\[嘻嘻\]|\[赞\]|\[酷\]|\[鼓掌\]|\[馋嘴\]|\[爱你\]|\[good\]|\[心\]|\[花心\]|\[可爱\]|\[威武\]|\[害羞\]|\[给力\]|\[蛋糕\]|\[耶\]|\[太开心\]|\[亲亲\]|\[笑哈哈\]|\[太阳\]|\[抱抱\]|\[带感\]|\[爱心传递\]|\[做鬼脸\]|\[握手\]|\[干杯\]|\[ok\]|\[得意地笑\]|\[礼物\]|\[阳光\]|\[咖啡\]|\[来\]|\[偷乐\]|\[愛你\]|\[帅\]|\[萌\]|\[好激动\]|\[飞个吻\]|\[乐乐\])",
    "neg_emoji": "(\[泪\]|\[汗\]|\[衰\]|\[怒\]|\[抓狂\]|\[挖鼻屎\]|\[可怜\]|\[晕\]|\[生病\]|\[呵呵\]|\[哼\]|\[委屈\]|\[疑问\]|\[鄙视\]|\[睡觉\]|\[蜡烛\]|\[打哈气\]|\[黑线\]|\[悲伤\]|\[失望\]|\[伤心\]|\[阴险\]|\[怒骂\]|\[困\]|\[囧\]|\[挤眼\]|\[吐\]|\[弱\]|\[右哼哼\]|\[懒得理你\]|\[闭嘴\]|\[猪头\]|\[泪流满面\]|\[淚\]|\[左哼哼\])",
    "neu_emoji": "(\[话筒\]|\[思考\]|\[吃惊\]|\[兔子\]|\[围观\]|\[奥特曼\]|\[熊猫\]|\[微风\]|\[浮云\]|\[月亮\]|\[雪\]|\[嘘\]|\[神马\]|\[转发\]|\[照相机\]|\[钱\]|\[飞机\]|\[手套\]|\[落叶\]|\[下雨\]|\[钟\]|\[汽车\]|\[自行车\]|\[雪人\]|\[音乐\])"
}
emoji_type2id = {"pos_emoji": 0, "neg_emoji": 1, "neu_emoji": 2}

def download_picture(url, save_dir):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    response = requests.get(url, headers=headers)
    with open(save_dir, 'wb') as f:
        f.write(response.content)


def str_to_time(time_str):
    # time_str format: Tue Nov 27 11:31:39 +0800 2012
    # return format: Year, Month, Day, Day of the week, Hour, Minute, Second
    val = time_str.split(" ")
    clock = val[3].split(":")
    return int(val[-1]), str_to_month[val[1]], int(val[2]), str_to_day_of_the_week[val[0]], int(clock[0]), int(clock[1]), int(clock[2])


def filter_text(text, filter_emoticon=True):
    # filter @, links, emojis, and special characters
    text = re.sub("//@.*:", "", text)
    text = re.sub("@.*\s", "", text)
    text = re.sub("★*http.*([a-z]|[A-Z]|[0-9])", "", text)
    text = re.sub("★|#|\n|\r", "", text)
    emoticons_count = None
    if filter_emoticon:
        emoticons_count = [0, 0, 0]
        for type in emojis:
            emoticon_pattern = re.compile(emojis[type])
            for elem in emoticon_pattern.findall(text):
                emoticons_count[emoji_type2id[type]] += 1
            text = re.sub(emojis[type], "", text)
    return text, emoticons_count


def convert_image_to_tensor(path):
    # currently ignore microblog images since they are too large
    img = cv2.imread(path)
    trans = transforms.ToTensor()
    return trans(img)


def generate_masks(pos, max_len):
    mask = np.zeros((len(pos) - 1, max_len))
    for i in range(len(pos) - 1):
        for j in range(pos[i], pos[i + 1]):
            mask[i][j] = 1 / max(pos[i + 1] - pos[i], 1)
    return torch.FloatTensor(mask)


def make_input(batch):
    return {
        "features_x": batch[0],
        "user_text_x" : batch[1],  
        "user_text_att_mask": batch[2],
        "microblog_text_x":  batch[3],
        "microblog_text_att_mask": batch[4],
        "user_avatar_x" : batch[5],
        "microblog_image_x": batch[6],
        "microblog_image_mask": batch[7],
        "y" : batch[8],
        "user_id": batch[9]
    }