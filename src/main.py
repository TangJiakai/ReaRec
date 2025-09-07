# -*- coding: UTF-8 -*-

from datetime import datetime
import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch

sys.path.append("..")

from helpers import *
from models import *
from utils import utils


def parse_global_args(parser):
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Set CUDA_VISIBLE_DEVICES, default for CPU only",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=logging.INFO,
        help="Logging Level, 0, 10, ..., 50",
    )
    parser.add_argument("--log_file", type=str, default="", help="Logging file path")
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed of numpy and pytorch"
    )
    parser.add_argument(
        "--load", type=int, default=0, help="Whether load model and continue to train"
    )
    parser.add_argument(
        "--load_model_path", type=str, default=None, help="Model load path"
    )
    parser.add_argument(
        "--train", type=int, default=1, help="To train the model or not."
    )
    parser.add_argument(
        "--regenerate",
        type=int,
        default=0,
        help="Whether to regenerate intermediate files",
    )
    return parser


def main():
    logging.info("-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45)
    exclude = [
        "check_epoch",
        "log_file",
        "model_path",
        "path",
        "pin_memory",
        "load",
        "regenerate",
        "sep",
        "train",
        "verbose",
        "metric",
        "test_epoch",
        "buffer",
    ]
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    utils.init_seed(args.random_seed)

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cpu")
    if args.gpu != "" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    logging.info("Device: {}".format(args.device))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + ".pkl")
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info("Load corpus from {}".format(corpus_path))
        corpus = pickle.load(open(corpus_path, "rb"))
    else:
        corpus = reader_name(args)
        logging.info("Save corpus to {}".format(corpus_path))
        pickle.dump(corpus, open(corpus_path, "wb"))

    # Define model
    model = model_name(args, corpus).to(args.device)
    logging.info("#params: {}".format(model.count_variables()))
    logging.info(model)

    # Define dataset
    data_dict = dict()
    for phase in ["train", "valid", "test"]:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()

    # Run model
    runner = runner_name(args)
    if args.load > 0:
        model.load_model(args.load_model_path)
    logging.info("Valid Before Training: " + runner.print_res(data_dict["valid"]))
    logging.info("Test Before Training: " + runner.print_res(data_dict["test"]))
    if args.train > 0:
        runner.train(data_dict)

    # Evaluate final results
    eval_res = runner.print_res(data_dict["test"])
    logging.info(os.linesep + "Test After Training: " + eval_res)
    model.actions_after_train()
    logging.info(os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45)


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description="Model")
    init_parser.add_argument(
        "--model_name", type=str, default="SASRec", help="Choose a model to run."
    )
    init_args, init_extras = init_parser.parse_known_args()

    model_name = eval("{0}.{0}".format(init_args.model_name))
    reader_name = eval("{0}.{0}".format(model_name.reader))  # model chooses the reader
    runner_name = eval("{0}.{0}".format(model_name.runner))  # model chooses the runner

    # Args
    parser = argparse.ArgumentParser(description="")
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ["lr", "l2"] + model_name.extra_log_args:
        log_args.append(arg + "=" + str(eval("args." + arg)))
    log_file_name = "__".join(log_args).replace(" ", "__")
    # add a time stamp to the log file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + str(
        datetime.now().microsecond
    ).zfill(6)
    log_file_name = log_file_name + "__" + timestamp
    if args.log_file == "":
        args.log_file = "log/{}/{}.log".format(init_args.model_name, log_file_name)
    if args.model_path == "":
        args.model_path = "save_model/{}/{}.pt".format(
            init_args.model_name, log_file_name
        )
    args.model_name = init_args.model_name

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
