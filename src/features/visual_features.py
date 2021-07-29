import numpy as np
import os.path as osp
import cv2

class VisualFeatures():
    def __init__(self):
        super().__init__()
        self.exists = False
        self.saturation = np.zeros(2)
        self.brightness = np.zeros(2)
        self.warm_color = 0
        self.clean_color = 0
        self.five_color_theme = np.zeros(15)

    def extract_features(self, image_path):
        if image_path is not None and osp.exists(image_path):
            self.exists = True
            rgb_img = cv2.imread(image_path)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.split(hsv_img)
            self.five_color_theme
            color_cnt = {}
            for i in range(0, 224):
                for j in range(0, 224):
                    color_cnt[(H[i][j],S[i][j],V[i][j])] = color_cnt.get((H[i][j],S[i][j],V[i][j]),0) + 1
            color_cnt = sorted(color_cnt.items(), key=lambda d:d[1], reverse=True)
            self.five_color_theme[0] = color_cnt[0][0][0]
            self.five_color_theme[1] = color_cnt[0][0][1]
            self.five_color_theme[2] = color_cnt[0][0][2]
            self.five_color_theme[3] = color_cnt[1][0][0]
            self.five_color_theme[4] = color_cnt[1][0][1]
            self.five_color_theme[5] = color_cnt[1][0][2]
            self.five_color_theme[6] = color_cnt[2][0][0]
            self.five_color_theme[7] = color_cnt[2][0][1]
            self.five_color_theme[8] = color_cnt[2][0][2]
            self.five_color_theme[9] = color_cnt[3][0][0]
            self.five_color_theme[10] = color_cnt[3][0][1]
            self.five_color_theme[11] = color_cnt[3][0][2]
            self.five_color_theme[12] = color_cnt[4][0][0]
            self.five_color_theme[13] = color_cnt[4][0][1]
            self.five_color_theme[14] = color_cnt[4][0][2]
            ## 明度（V）
            average_v  = np.average(V)
            var_v = np.var(V)
            self.brightness[0] = average_v
            self.brightness[1] = var_v
            ## 饱和度（S）
            average_s  = np.average(S)
            var_s = np.var(S)
            self.saturation[0] = average_s
            self.saturation[1] = var_s
            ## 色调（H）
            S = S.flatten()
            H = H.flatten()
            warm_cnt = len(np.intersect1d(np.where(H>=30), np.where(H<=110)))
            self.warm_color = warm_cnt / 224 / 224
            clean_cnt = len(np.where(S<0.7*255))
            self.clean_color = clean_cnt / 224 /224
            

def compress_visual_features(features):
    compressed = VisualFeatures()
    count = 0
    for feat in features:
        if feat.exists:
            count += 1
            compressed.exists = True
            compressed.saturation += feat.saturation
            compressed.brightness += feat.brightness
            compressed.warm_color += feat.warm_color
            compressed.clean_color += feat.clean_color
            compressed.five_color_theme += feat.five_color_theme
    compressed.saturation /= count
    compressed.brightness /= count
    compressed.warm_color /= count
    compressed.clean_color /= count
    compressed.five_color_theme /= count
    return compressed
