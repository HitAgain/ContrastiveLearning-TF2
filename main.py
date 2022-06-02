# coding=utf-8
# author:HitAgain
# time:2022/03/08
# main.py

import argparse
import logging
import os

from train import SimCse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="train text file")
    parser.add_argument("--valid_file", type=str, default="./data/dev.txt", help="dev text file")
    parser.add_argument("--pretrain_dir", type=str, default="./chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--export_path", type=str, default="./serving", help="serve model output path")
    parser.add_argument("--epochs", type=int, default=8, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--attention_dropout_rate", type=float, default=0.3, help="attention_dropout_rate")
    parser.add_argument("--hidden_dropout_rate", type=float, default=0.3, help="hidden_dropout_rate")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    SimCse = SimCse(args)
    SimCse.train()
    SimCse.export()