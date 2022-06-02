# coding=utf-8
# author:HitAgain
# time:2022/03/08
# SimCse训练类

import sys
import codecs
import logging
import json
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import tensorflow as tf
from transformers import AutoTokenizer

from unsupvise import UnsupvisedSimCSE
from supvise import SupvisedSimCSE
from data import GeneratorForSimCSE

from loss import SimLoss


class SimCse(object):
    def __init__(self, args):
        self.args = args
        self.SimCSE = None
        if self.args.mode == "unsup":
            self.SimCSE = UnsupvisedSimCSE(self.args)
            logging.info("[success]UnsupvisedSimCSE model build")
        else:
            self.SimCSE = SupvisedSimCSE(self.args)
            logging.info("[success]SupvisedSimCSE model build")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_dir)

    def _build_train_data(self, file_path):
        if self.args.mode = "unsup":
            train_input_ids = []
            train_token_type_ids = []
            train_attention_mask = []
            with open(file_path, "r", encoding="utf-8") as fin:
                for sample in fin:
                  try:
                    sentence = sample.strip("\n").split("\t")
                    tokenize_result_dic = tokenizer(sentence)
                    train_input_ids.append(tokenize_result_dic["input_ids"])
                    train_token_type_ids.append(tokenize_result_dic["token_type_ids"])
                    train_attention_mask.append(tokenize_result_dic["attention_mask"])
                  except Exception as e:
                    logging.error("data load error")
            return GeneratorForSimCSE(train_input_ids, train_token_type_ids, train_attention_mask, self.args.batch_size)
        else:
            pass

    def train(self):
        if self.mode = "unsupvise":
            train_generator = self._build_train_data(self.args.train_file_path)
            valid_generator = self._build_train_data(self.args.valid_file_path)
            logging.info("[success]data prepared")
            self.SimCSE.compile(loss= tf.keras.losses.categorical_crossentropy(from_logits=True),
                               optimizer="adam",
                               metrics=['accuracy'])
            self.SimCSE.fit(
                x = train_generator,
                validation_data = valid_generator,
                epochs = self.args.epochs,
                steps_per_epoch=len(train_generator),
                verbose = 1,
                callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.005, mode='max')] 
            )
        else:
            train_generator = self._build_train_data(self.args.train_file_path)
            valid_generator = self._build_train_data(self.args.valid_file_path)
            logging.info("[success]data prepared")
            self.SimCSE.compile(loss= SimLoss(name="sim_loss"),
                               optimizer="adam",
                               metrics=['accuracy'])
            self.SimCSE.fit(
                x = train_generator,
                validation_data = valid_generator,
                epochs = self.args.epochs,
                steps_per_epoch=len(train_generator),
                verbose = 1,
                callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.005, mode='max')] 
            )

    def export(self):
        tf.saved_model.save(self.SimCSE, self.args.export_path, signatures = {"sentenceEmbedding": self.SimCSE.embedding})
