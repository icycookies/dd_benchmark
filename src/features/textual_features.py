import csv
from typing import Dict, List

import numpy as np

from .features import Features

CACHED_DATA: Dict[str, List[int]] = {}


class TextualFeatures(Features):
    def __init__(self):
        super().__init__()
        self.emotional_words_count = np.zeros(2)
        self.emoticon_count = np.zeros(3)
        self.pronoun_count = np.zeros(3)
        self.punctuation_count = np.zeros(3)
        # biology, body, health, death, society, family, friends, money, work and leisure
        self.topic_related_words_count = np.zeros(10)

    def extract_features(self, user, cache_file):
        if cache_file is not None:
            try:
                val = CACHED_DATA[user.id]
            except KeyError:
                with open(cache_file, "r") as f:
                    table = csv.reader(f)
                    for idx, row in enumerate(table):
                        if idx == 0:
                            continue
                        CACHED_DATA[row[0]] = [float(row[i]) for i in range(1, 103)]
                val = CACHED_DATA[user.id]
            num_words = val[91]
            self.emotional_words_count = np.array(val[38:40]) * num_words
            self.emoticon_count = np.array(user.emoticon_count)
            self.pronoun_count = np.array([val[3] + val[4], val[5], val[6] + val[7]]) * num_words
            self.punctuation_count = np.array(val[84:86] + [val[80] / 3]) * num_words
            self.topic_related_words_count = np.array(val[56:59] + val[71:72] + val[33:36] + val[69:70] + val[65:66] + val[67:68]) * num_words
