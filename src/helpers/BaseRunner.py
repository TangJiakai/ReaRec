# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import pickle

from utils import utils
from utils.constants import *
from utils.metrics import *
from models.BaseModel import BaseModel


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument("--epoch", type=int, default=300, help="Number of epochs.")
        parser.add_argument(
            "--check_epoch",
            type=int,
            default=0,
            help="Check some tensors every check_epoch.",
        )
        parser.add_argument(
            "--test_epoch",
            type=int,
            default=-1,
            help="Print test results every test_epoch (-1 means no print).",
        )
        parser.add_argument(
            "--early_stop",
            type=int,
            default=10,
            help="The number of epochs when valid results drop continuously.",
        )
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
        parser.add_argument(
            "--l2", type=float, default=0, help="Weight decay in optimizer."
        )
        parser.add_argument(
            "--batch_size", type=int, default=2048, help="Batch size during training."
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1024,
            help="Batch size during testing.",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="optimizer: SGD, Adam, Adagrad, Adadelta",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=10,
            help="Number of processors when prepare batches in DataLoader",
        )
        parser.add_argument(
            "--pin_memory", type=int, default=0, help="pin_memory in DataLoader"
        )
        parser.add_argument(
            "--topk",
            type=str,
            default="5,10,20,50,100",
            help="The number of items recommended to each user.",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="NDCG,HR,MRR",
            help="metrics: NDCG, HR, MRR, PRECISION, MAP",
        )
        parser.add_argument(
            "--main_metric",
            type=str,
            default="NDCG@20",
            help="Main metric to determine the best model.",
        )
        parser.add_argument(
            "--clip_grad",
            action="store_true",
            help="Whether to use gradient clip or not.",
        )
        parser.add_argument(
            "--max_norm",
            type=float,
            default=1.0,
            help="Max norm of the gradients for gradient clipping",
        )
        return parser

    @staticmethod
    def evaluate_method(
        predictions: np.ndarray, targets: np.ndarray, topk: list, metrics: list
    ) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param targets: (-1,) shape, the ground-truth item id
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()

        # torch version of topk
        max_topk = max(topk)
        batch_size = 65536
        top_indices_list = []
        for start in range(0, predictions.shape[0], batch_size):
            end = min(start + batch_size, predictions.shape[0])
            batch = predictions[start:end]
            _, topk_idx = torch.topk(batch, k=max_topk, dim=1)
            top_indices_list.append(topk_idx.cpu().numpy())
            del batch, topk_idx
            torch.cuda.empty_cache()
        top_idx = np.vstack(top_indices_list)

        targets = targets.cpu().numpy()
        pos_matrix = top_idx == targets.reshape(-1, 1)
        for m in metrics:
            m_res = eval(m)(pos_matrix, topk)
            for k, res in zip(topk, m_res):
                evaluations[m + "@" + str(k)] = round(res.mean(), 4)

        return evaluations

    def __init__(self, args):
        self.train_models = args.train
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(",")]
        self.metrics = [m.strip().upper() for m in args.metric.split(",")]
        self.main_metric = args.main_metric  # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.clip_grad = args.clip_grad
        self.max_norm = args.max_norm
        self.device = args.device

        self.time = None  # will store [start_time, last_step_time]

        self.log_path = os.path.dirname(args.log_file)  # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[
            0
        ]  # appendix for prediction saving

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info("Optimizer: " + self.optimizer_name)
        optimizer = eval("torch.optim.{}".format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        return optimizer

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict["train"].model
        main_metric_results, valid_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict["train"], epoch=epoch + 1)
                if np.isnan(loss):
                    logging.info("Loss is Nan. Stop training at %d." % (epoch + 1))
                    break
                training_time = self._check_time()

                # Observe selected tensors
                if (
                    len(model.check_list) > 0
                    and self.check_epoch > 0
                    and epoch % self.check_epoch == 0
                ):
                    utils.check(model.check_list)

                # Record valid results
                valid_result = self.evaluate(
                    data_dict["valid"], [self.main_topk], self.metrics
                )
                valid_results.append(valid_result)
                main_metric_results.append(valid_result[self.main_metric])
                logging_str = "Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	valid=({})".format(
                    epoch + 1, loss, training_time, utils.format_metric(valid_result)
                )

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(
                        data_dict["test"], self.topk[:1], self.metrics
                    )
                    logging_str += " test=({})".format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += " [{:<.1f} s]".format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or (
                    hasattr(model, "stage") and model.stage == 1
                ):
                    model.save_model()
                    logging_str += " *"
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info(
                        "Early stop at %d based on valid result." % (epoch + 1)
                    )
                    break

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith("y"):
                logging.info(
                    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
                )
                exit(1)

        # Find the best valid result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(
            os.linesep
            + "Best Iter(valid)={:>5}\t valid=({}) [{:<.1f} s] ".format(
                best_epoch + 1,
                utils.format_metric(valid_results[best_epoch]),
                self.time[1] - self.time[0],
            )
        )
        model.load_model()

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )
        for batch in tqdm(
            dl, leave=False, desc="Epoch {:<3}".format(epoch), ncols=100, mininterval=1
        ):
            batch = utils.batch_to_gpu(batch, model.device)

            model.optimizer.zero_grad()
            out_dict = model(batch, epoch=epoch)

            loss = model.loss(out_dict)
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(
            criterion[-self.early_stop :]
        ):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def evaluate(
        self, dataset: BaseModel.Dataset, topks: list, metrics: list
    ) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions, item_ids = self.predict(dataset)
        return self.evaluate_method(predictions, item_ids, topks, metrics)

    def predict(self, dataset: BaseModel.Dataset) -> np.ndarray:
        dataset.model.eval()
        predictions, item_ids = list(), list()
        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc="Predict"):
            if hasattr(dataset.model, "inference"):
                prediction = dataset.model.inference(
                    utils.batch_to_gpu(batch, dataset.model.device)
                )["prediction"]
            else:
                out_dict = dataset.model(
                    utils.batch_to_gpu(batch, dataset.model.device), stage="infer"
                )
                prediction = out_dict["prediction"]
            predictions.append(prediction.detach())
            item_ids.append(batch[ITEM_ID].detach())
        predictions = torch.cat(predictions, dim=0)
        predictions[:, -1] = -torch.inf
        item_ids = torch.cat(item_ids, dim=0)
        return predictions, item_ids

    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        res_str = "(" + utils.format_metric(result_dict) + ")"
        return res_str
