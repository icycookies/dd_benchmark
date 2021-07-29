import json
import numpy as np
import torch

from .user_features import UserFeatures
from .microblog_features import MicroblogFeatures, compress_microblog_features

def _generate_features(feat, used_feat_dict):
    feat = feat.__dict__
    feat_arr = np.array([])
    for feat_name in used_feat_dict:
        if isinstance(used_feat_dict[feat_name], int):
            if used_feat_dict[feat_name] == 1:
                tmp = np.array([feat[feat_name]])
            else:
                tmp = np.array(feat[feat_name])
        else:
            tmp = _generate_features(feat[feat_name], used_feat_dict[feat_name])
        feat_arr = np.concatenate((feat_arr, tmp))
    return feat_arr

def generate_predefined_features(user, 
                                 microblogs, 
                                 cache_file="../data/weibo2012/processed",
                                 generate_microblog_features=False, 
                                 concatenate_microblog_features=False,
    ):
    # --- Input Format ---
    # user
    # microblogs: Dict, { User.id -> [ Microblogs ] }
    # generate_microblog_features: Bool, genenrate what level features
    # --- Output Format ---
    # user_feat: FloatTensor, [feat_dim]
    # microblog_feat: Option<FloatTensor>, [max_microblogs, feat_dim]
    used_feat = json.load(open("./features/used_features.json", "r"))
    user_feat = UserFeatures()
    user_feat.extract_features(user, microblogs, cache_file)
    user_feat_tensor = torch.FloatTensor(_generate_features(user_feat, used_feat['user']))
        
    if generate_microblog_features:
        if not concatenate_microblog_features:
            microblog_feat_arr = None
            for microblog in microblogs:
                microblog_feat = MicroblogFeatures()
                microblog_feat.extract_features(microblog)
                tmp = generate_microblog_features(microblog_feat, used_feat['microblog'])
                if microblog_feat_arr is None:
                    microblog_feat_arr = np.array(tmp)
                else:
                    microblog_feat_arr = np.concatenate((microblog_feat_arr, tmp), axis=0)
            microblog_feat_tensor = torch.FloatTensor(microblog_feat_arr)
            return user_feat_tensor, microblog_feat_tensor
        else:
            microblogs_feat_list = []
            for microblog in microblogs:
                microblog_feat = MicroblogFeatures()
                microblog_feat.extract_features(microblog)
                microblogs_feat_list.append(microblog_feat)

            avg = compress_microblog_features(microblogs_feat_list)
            microblog_feat_tensor = torch.FloatTensor(generate_microblog_features(avg, used_feat['microblog']))
            user_feat_tensor = torch.cat((user_feat_tensor, microblog_feat_tensor))

    return user_feat_tensor    