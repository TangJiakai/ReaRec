# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import torch.nn as nn
import torch
from sklearn.decomposition import PCA

from utils.constants import *
from helpers.BaseReader import BaseReader


class LLMReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        BaseReader.parse_data_args(parser)
        parser.add_argument(
            "--plm_size",
            type=int,
            default=768,
            help="The embedding size of pretrained language model.",
        )
        parser.add_argument(
            "--plm_suffix",
            type=str,
            default="item_feature",
            help="The feature suffix of pretrained language model.",
        )
        return parser

    def __init__(self, args):
        super(LLMReader, self).__init__(args)
        self.plm_size = args.plm_size
        self.plm_suffix = args.plm_suffix
        self.plm_emb = self._load_plm_embdding()

    def _load_plm_embdding(self):
        logging.info("Loading large language model embeddings...")
        feat_path = os.path.join(
            self.prefix, self.dataset, f"{self.dataset}.{self.plm_suffix}.npy"
        )
        load_feat = np.load(feat_path)
        pca = PCA(n_components=self.plm_size)
        load_feat = pca.fit_transform(load_feat)

        plm_emb = nn.Embedding(self.n_items, self.plm_size)
        plm_emb.weight.requires_grad = False
        plm_emb.weight.data[:-1].copy_(torch.from_numpy(load_feat))

        return plm_emb
