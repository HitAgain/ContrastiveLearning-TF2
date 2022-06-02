# coding=utf-8
# author:HitAgain
# time:2022/03/08
# data generator for SimCSE

import math
import numpy as np
import tensorflow as tf


class GeneratorForSimCSE(tf.keras.utils.Sequence):

    def __init__(self, feature_token_ids, feature_seg_ids, feature_mask_ids, batch_size):
        self.feature_token_ids = feature_token_ids
        self.feature_seg_ids = feature_seg_ids
        self.feature_mask_ids = feature_mask_ids
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.feature) / self.batch_size)

    def __getitem__(self, idx):
        batch_token_ids = self.feature_token_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_seg_ids = self.feature_seg_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_mask_ids = self.feature_mask_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_label_np = np.array(np.eye(self.batch_size)).astype(np.float32)
        batch_token_np = np.array(self.batch_pad(batch_token_ids))
        batch_seg_np = np.array(self.batch_pad(batch_seg_ids))
        batch_mask_np = np.array(self.batch_pad(batch_mask_ids))
        return [batch_token_np, batch_seg_np, batch_mask_np], batch_label_np

    def on_epoch_end(self):
        feature =  list(zip(self.feature_token_ids, self.feature_seg_ids, self.feature_mask_ids))
        np.random.shuffle(feature)
        self.feature_token_ids = [fb[0] for fb in feature]
        self.feature_seg_ids = [fb[1] for fb in feature]
        self.feature_mask_ids = [fb[2] for fb in feature]

    def batch_pad(self, x):
        ml = max([len(i) for i in x])
        return [i + [0] * (ml-len(i)) for i in x]

class GeneratorForSupSimCSE(tf.keras.utils.Sequence):

    def __init__(self, feature_token_ids, feature_seg_ids, feature_mask_ids, batch_size):
        self.feature_token_ids = feature_token_ids
        self.feature_seg_ids = feature_seg_ids
        self.feature_mask_ids = feature_mask_ids
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.feature) / self.batch_size)

    def __getitem__(self, idx):
        batch_token_ids = self.feature_token_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_seg_ids = self.feature_seg_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_mask_ids = self.feature_mask_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        # label主要是用于占位
        batch_label_np = np.zeros((self.batch_size, 1))
        batch_token_np = np.array(self.batch_pad(batch_token_ids))
        batch_seg_np = np.array(self.batch_pad(batch_seg_ids))
        batch_mask_np = np.array(self.batch_pad(batch_mask_ids))
        return [batch_token_np, batch_seg_np, batch_mask_np], batch_label_np

    def on_epoch_end(self):
        feature =  list(zip(self.feature_token_ids, self.feature_seg_ids, self.feature_mask_ids))
        np.random.shuffle(feature)
        self.feature_token_ids = [fb[0] for fb in feature]
        self.feature_seg_ids = [fb[1] for fb in feature]
        self.feature_mask_ids = [fb[2] for fb in feature]

    def batch_pad(self, x):
        ml = max([len(i) for i in x])
        return [i + [0] * (ml-len(i)) for i in x]
