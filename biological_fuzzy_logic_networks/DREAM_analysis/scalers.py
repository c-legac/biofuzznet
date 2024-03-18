import numpy as np


class ClippingScaler:
    def __init__(self, Qhigh: float = 0.9, Qlow: float = 0.1):
        self.Qhigh = Qhigh
        self.Qlow = Qlow

    def fit(self, data):
        self.q_high = data.quantile(self.Qhigh)
        self.q_low = data.quantile(self.Qlow)

    def fit_transform(self, data):
        self.fit(data)
        clip_norm = self.transform(data)

        return clip_norm

    def transform(self, data):
        clip_norm = data[data.lt(self.q_high, axis=1)]
        clip_norm = clip_norm.fillna(clip_norm.max())
        clip_norm = clip_norm[clip_norm.gt(self.q_low, axis=1)]
        clip_norm = clip_norm.fillna(clip_norm.min())

        return np.asarray(clip_norm)
