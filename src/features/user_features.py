import numpy as np
import time
import os.path as osp

from .features import Features
from .visual_features import VisualFeatures
from .textual_features import TextualFeatures

def get_timestamp(time_str):
    return time.mktime(time.strptime(time_str, "%a %b %d %H:%M:%S +0800 %Y"))

class UserFeatures(Features):
    def __init__(self):
        super().__init__()
        self.gender = 0
        self.name_length = 0
        self.all_microblogs_count = 0
        self.original_microblogs_ratio = 0
        self.microblog_with_image_ratio = 0
        self.tweet_time_count = np.zeros(24)
        self.late_post_ratio = 0
        self.post_frequency = 0
        self.time_deviation = 0
        self.followers = np.zeros(3)
        self.avatar_features = None
        self.textual_features = None

    def extract_features(self, user, microblogs, cache_file):
        # input format:
        # user: User
        # microblogs: List, [ Microblog ]
        self.gender = user.gender
        self.name_length = len(user.name)
        self.all_microblogs_count = len(microblogs)
        self.followers = np.array([user.followers_count, user.bi_followers_count, user.friends_count])

        original_microblogs_count = 0
        microblog_with_image_count = 0
        late_post_count = 0
        max_timestamp = 0
        min_timestamp = np.inf
        average_timestamp = 0
        for microblog in microblogs:
            timestamp = get_timestamp(microblog.time_str)
            if timestamp > max_timestamp:
                max_timestamp = timestamp
            if timestamp < min_timestamp:
                min_timestamp = timestamp
            if not microblog.retweet:
                original_microblogs_count += 1
            if microblog.image_path is not None:
                microblog_with_image_count += 1
            if microblog.hour < 6 or microblog.hour >= 23:
                late_post_count += 1
            self.tweet_time_count[microblog.hour] += 1
            average_timestamp += timestamp

        self.original_microblogs_ratio = original_microblogs_count / self.all_microblogs_count
        self.microblog_with_image_ratio = microblog_with_image_count / self.all_microblogs_count
        self.late_post_ratio = late_post_count / self.all_microblogs_count
        num_weeks = (max_timestamp - min_timestamp) / 604800
        self.post_frequency = self.all_microblogs_count / num_weeks if num_weeks != 0 else 1
        
        average_timestamp /= self.all_microblogs_count
        tmp = 0
        for microblog in microblogs:
            timestamp = get_timestamp(microblog.time_str)
            tmp += ((timestamp - average_timestamp) / 3600) ** 2
        self.time_deviation = (tmp / self.all_microblogs_count) ** 0.5

        self.avatar_features = VisualFeatures()
        self.avatar_features.extract_features(user.avatar_path)

        self.textual_features = TextualFeatures()
        self.textual_features.extract_features(user, osp.join(cache_file, "WenxinFeat.csv"))
