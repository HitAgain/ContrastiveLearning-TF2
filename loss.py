# coding=utf-8
# author:HitAgain
# time:2022/03/08
# 有监督SimCse的自定义损失函数


import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="simcse")
class SimLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SimLoss, self).__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        # 构造y_true
        idxs = tf.range(0, tf.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.float32)
        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)

    def get_config(self):
        return super(SimLoss, self).get_config()
