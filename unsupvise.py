# coding=utf-8
# author:HitAgain
# time:2022/03/08

import sys
import codecs
import logging

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import tensorflow as tf
from transformers import BertConfig, TFBertModel, AutoTokenizer


class UnsupvisedSimCSE(tf.keras.Model):
    def __init__(self, args):
        super(UnsupvisedSimCSE, self).__init__()
        conf = BertConfig.from_pretrained(args.pretrain_dir)
        conf.attention_probs_dropout_prob = args.attention_dropout_rate
        conf.hidden_dropout_prob = args.hidden_dropout_rate
        self.encoder = TFBertModel.from_pretrained(args.pretrain_dir, config=conf)

    def call(self, inputs, training=True):
        input_ids, token_type_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        pool_out_1 = self.encoder(input_ids, token_type_ids, attention_mask)[1]
        pool_out_2 = self.encoder(input_ids, token_type_ids, attention_mask)[1]
        p_norm1 = tf.math.l2_normalize(pool_out_1, axis=1)
        p_norm2 = tf.math.l2_normalize(pool_out_2, axis=1)
        similarities = tf.matmul(y_pred_1, y_pred_2, transpose_b=True)
        similarities = similarities * 20
        return {"similarities":similarities}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='input_ids'),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='token_type_ids'),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='attention_mask'),
    ])
    def embedding(self, input_ids, token_type_ids, attention_mask, training=False):
        pool_out = self.encoder(input_ids, token_type_ids, attention_mask)[1]
        final_out = tf.math.l2_normalize(pool_out, axis=1)
        return {"predict_representation":final_out}
