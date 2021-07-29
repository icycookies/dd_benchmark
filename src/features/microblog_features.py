from .features import Features
from .visual_features import VisualFeatures, compress_visual_features

class MicroblogFeatures(Features):
    def __init__(self):
        super().__init__()
        self.text_length = 0
        self.image_features = None

    def extract_features(self, microblog):
        self.text_length = len(microblog.text)
        self.image_features = VisualFeatures()
        self.image_features.extract_features(microblog.image_path)

def compress_microblog_features(features):
    compressed = MicroblogFeatures()
    for feat in features:
        compressed.text_length += feat.text_length
    compressed.text_length /= len(features)

    compressed.image_features = compress_visual_features([feat.image_features for feat in features])
    return compressed