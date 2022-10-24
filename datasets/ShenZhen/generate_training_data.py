import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from data.transform import standard_transform


def ShenZhen_generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of METRLA dataset:
        - Normalization method: standard norm.
        - Dataset division: 7:1:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic speed, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    dataset_name = args.dataset_name
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day

    # read data
    df = pd.read_csv(data_file_path)
    data = np.expand_dims(df.values, axis=-1)

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
        time_ind = [i % steps_per_day / 96 for i in range(data.shape[0])]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_of_week:
        day_in_week = [(i // 96) % 7 for i in range(data.shape[0])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, data.shape[1], 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

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
    adj_mx = pd.read_csv(graph_file_path)
    with open("./datasets/{0}/output/in{1}_out{2}/adj_ShenZhen.pkl".format(dataset_name,history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(adj_mx, f)



