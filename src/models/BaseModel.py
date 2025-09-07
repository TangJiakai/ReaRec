# -*- coding: UTF-8 -*-

import torch
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from typing import List
import copy
import random

from utils import utils
from helpers.BaseReader import BaseReader
from utils.constants import *


class BaseModel(nn.Module):
    reader, runner = None, None  # choose helpers in specific model classes
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--model_path", type=str, default="", help="Model save path."
        )
        parser.add_argument(
            "--attention_path", type=str, default="", help="Attention save path."
        )
        parser.add_argument(
            "--buffer",
            type=int,
            default=1,
            help="Whether to buffer feed dicts for valid/test",
        )
        return parser

    @staticmethod
    def init_weights(m):
        if "Linear" in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif "Embedding" in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus: BaseReader):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.buffer = args.buffer
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch

    """
	Key Methods
	"""

    def _define_params(self):
        pass

    def forward(self, feed_dict: dict) -> dict:
        """
        :param feed_dict: batch prepared in Dataset
        :return: out_dict, including prediction with shape [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict) -> torch.Tensor:
        pass

    """
	Auxiliary Methods
	"""

    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if "bias" in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{"params": weight_p}, {"params": bias_p, "weight_decay": 0}]
        return optimize_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info("Load model from " + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_after_train(self):  # e.g., save selected parameters
        pass

    """
	Define Dataset Class
	"""

    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase  # train / valid / test

            self.buffer_dict = dict()
            self.data = corpus.data_dict[phase]

        def __len__(self):
            if type(self.data) == dict:
                for key in self.data:
                    return len(self.data[key])
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            if self.model.buffer and self.phase != "train":
                return self.buffer_dict[index]
            return self._get_feed_dict(index)

        # ! Key method to construct input data for a single instance
        def _get_feed_dict(self, index: int) -> dict:
            pass

        # Called after initialization
        def prepare(self):
            if self.model.buffer and self.phase != "train":
                for i in tqdm(
                    range(len(self)), leave=False, desc=("Prepare " + self.phase)
                ):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        # Called before each training epoch (only for the training dataset)
        def actions_before_epoch(self):
            pass

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            feed_dict = dict()
            for key in feed_dicts[0]:
                feed_dict[key] = torch.stack([d[key] for d in feed_dicts])
            feed_dict["batch_size"] = len(feed_dicts)
            feed_dict["phase"] = self.phase
            return feed_dict


class SequentialModel(BaseModel):
    reader, runner = "BaseReader", "BaseRunner"

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--test_all", type=int, default=1, help="Whether testing on all the items."
        )
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.dropout = args.dropout
        self.test_all = args.test_all

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            item_seq, target_item, item_seq_len = (
                self.data[ITEM_SEQ][index],
                self.data[ITEM_ID][index],
                self.data[ITEM_SEQ_LEN][index],
            )
            feed_dict = {
                ITEM_SEQ: item_seq,
                ITEM_ID: target_item,
                ITEM_SEQ_LEN: item_seq_len,
            }
            return feed_dict

