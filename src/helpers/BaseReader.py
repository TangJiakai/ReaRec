# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

from utils import utils
from utils.constants import *


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument(
            "--path",
            type=str,
            default="../../datasets/processed",
            help="Input data dir.",
        )
        parser.add_argument(
            "--dataset", type=str, default="Yelp", help="Choose a dataset."
        )
        parser.add_argument("--sep", type=str, default=",", help="sep of csv file.")
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

    def _read_data(self):
        logging.info(
            'Reading data from "{}", dataset = "{}" '.format(self.prefix, self.dataset)
        )
        data_df = dict()
        for key in ["train", "valid", "test"]:
            data_df[key] = pd.read_csv(
                os.path.join(
                    self.prefix, self.dataset, f"{self.dataset}.{key}.remap.csv"
                ),
                sep=self.sep,
            ).reset_index(drop=True)
            data_df[key] = utils.eval_list_columns(data_df[key])

        logging.info("Counting dataset statistics...")
        key_columns = [USER_ID, ITEM_ID, ITEM_SEQ]
        all_df = pd.concat(
            [data_df[key][key_columns] for key in ["train", "valid", "test"]],
            ignore_index=True,
        )
        self.n_users = all_df[USER_ID].nunique() + 1  # including [PAD]
        self.n_items = (
            max(
                all_df[ITEM_ID].values.max(), all_df[ITEM_SEQ].agg(np.concatenate).max()
            )
            + 2
        )  # including [PAD]
        logging.info(
            '"# user": {}, "# item": {}, "# entry": {}'.format(
                self.n_users - 1, self.n_items - 1, len(all_df)
            )
        )
        all_df[ITEM_SEQ] = all_df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])
        all_df[ITEM_SEQ_LEN] = all_df[ITEM_SEQ].apply(len)

        all_data_dict = dict()
        all_data_dict[USER_ID] = torch.from_numpy(all_df[USER_ID].values).long()
        all_data_dict[ITEM_ID] = torch.from_numpy(all_df[ITEM_ID].values).long()
        item_seq = [
            torch.from_numpy(np.array(x)).long() for x in all_df[ITEM_SEQ].values
        ]

        # Left Padding
        left_padded_seqs = [
            pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=self.n_items - 1)
            for seq in item_seq
        ]
        all_data_dict[ITEM_SEQ] = torch.stack(left_padded_seqs)

        all_data_dict[ITEM_SEQ_LEN] = torch.from_numpy(
            all_df[ITEM_SEQ_LEN].values
        ).long()

        self.data_dict = {key: dict() for key in ["train", "valid", "test"]}
        start, end = 0, 0
        for key in ["train", "valid", "test"]:
            end = start + len(data_df[key][ITEM_ID])
            for c in all_data_dict:
                self.data_dict[key][c] = all_data_dict[c][start:end]
            start = end

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")