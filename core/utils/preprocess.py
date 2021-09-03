import numpy as np


def proc_bezier_feat(feat, bez_feat_pad_size):

    if feat.shape[0] > bez_feat_pad_size:
        feat = feat[:bez_feat_pad_size]

    feat = np.pad(
        feat,
        ((0, bez_feat_pad_size - feat.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return feat
