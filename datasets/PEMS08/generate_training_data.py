import os
import pathlib
import sys
import shutil
import pickle
import argparse

import networkx as nx
import numpy as np
import torch
from node2vec import Node2Vec

from datasets.PEMS08.generate_adj_mx import generate_adj_pems08
from utils import load_pkl

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from data.transform import standard_transform


def PEMS08_generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of PEMS04 dataset:
        - Normalization method: standard norm.
        - Dataset division: 6:2:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic flow, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    adj_dir = args.adj_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day

    # read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short +
                            valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # add external feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # dump data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data, f)
    # copy adj
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    if os.path.exists(graph_file_path):
        shutil.copyfile(graph_file_path, adj_dir + "/adj_mx.pkl")
    else:
        generate_adj_pems08()
        shutil.copyfile(graph_file_path, adj_dir + "/adj_mx.pkl")



