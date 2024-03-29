import pickle
import torch
import numpy as np


def data_transform(transform_type,data,args):
    if transform_type == "re_standard_transform":
        data_norm = re_standard_transform(data,**args)
    elif transform_type == "re_min_max_transform":
        data_norm = re_min_max_transform(data,**args)
    elif transform_type == "re_max_transform":
        data_norm = re_max_transform(data,**args)

    return data_norm

def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:
    data_train = data[:train_index[-1][1], ...]

    mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    return data_norm



def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:

    mean, std = kwargs["mean"], kwargs["std"]
    data = data * std
    data = data + mean
    return data



def min_max_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:

    data_train = data[:train_index[-1][1], ...]

    min_value = data_train.min(axis=(0, 1), keepdims=False)[0]
    max_value = data_train.max(axis=(0, 1), keepdims=False)[0]

    print("min: (training data)", min_value)
    print("max: (training data)", max_value)
    scaler = {}
    scaler["func"] = re_min_max_transform.__name__
    scaler["args"] = {"min_value": min_value, "max_value": max_value}
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        x = 1. * (x - min_value) / (max_value - min_value)
        x = 2. * x - 1.
        return x

    data_norm = normalize(data)
    return data_norm



def re_min_max_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:

    min_value, max_value = kwargs["min_value"], kwargs["max_value"]
    data = (data + 1.) / 2.
    data = 1. * data * (max_value - min_value) + min_value
    return data

def max_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:

    data_train = data[:train_index[-1][1], ...]

    max_value = data_train.max(axis=(0, 1), keepdims=False)[0]

    print("max: (training data)", max_value)
    scaler = {}
    scaler["func"] = re_max_transform.__name__
    scaler["args"] = {"max_value": max_value}
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        max_val = np.max(x)
        x = x / max_val
        return x

    data_norm = normalize(data)
    return data_norm


def re_max_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:

    max_value = kwargs["max_value"]
    data = data * max_value
    return data
