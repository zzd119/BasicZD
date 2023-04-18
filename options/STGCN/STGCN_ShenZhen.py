import argparse

import torch

from utils import load_adj


class STGCN_ShenZhen():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        adj_mx,_ = load_adj("datasets/ShenZhen/output/adj/adj_mx.pkl", "normlap")
        adj_mx = torch.tensor(adj_mx[0])
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--Ks", default=3)
        parser.add_argument("--Kt", default=3)
        parser.add_argument("--blocks", default=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]])
        parser.add_argument("--T", default=12)
        parser.add_argument("--n_vertex", default=156)
        parser.add_argument("--act_func", default="glu")
        parser.add_argument("--graph_conv_type", default="cheb_graph_conv")
        parser.add_argument("--gso", default=adj_mx)
        parser.add_argument("--bias", default=True)
        parser.add_argument("--droprate", default=0.5)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--weight_decay", default=0.0001)
        return parser
