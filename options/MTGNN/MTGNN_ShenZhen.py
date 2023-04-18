import argparse

import torch

from utils import load_adj


class MTGNN_ShenZhen():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        _, adj_mx = load_adj("datasets/ShenZhen/output/adj/adj_mx.pkl","doubletransition")
        adj_mx = torch.tensor(adj_mx) - torch.eye(325)
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--gcn_true", default=True)
        parser.add_argument("--buildA_true", default=True)
        parser.add_argument("--static_feat", default=None)
        parser.add_argument("--gcn_depth", default=2)
        parser.add_argument("--predefined_A", default=adj_mx)
        parser.add_argument("--dropout", default=0.3)
        parser.add_argument("--subgraph_size", default=20)
        parser.add_argument("--node_dim", default=40)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_node", default=156)
        parser.add_argument("--dilation_exponential", default=1)
        parser.add_argument("--conv_channels", default=32)
        parser.add_argument("--residual_channels", default=64)
        parser.add_argument("--skip_channels", default=64)
        parser.add_argument("--end_channels", default=128)
        parser.add_argument("--seq_length", default=12)
        parser.add_argument("--in_dim", default=2)
        parser.add_argument("--out_dim", default=12)
        parser.add_argument("--layers", default=3)
        parser.add_argument("--propalpha", default=0.05)
        parser.add_argument("--tanhalpha", default=3)
        parser.add_argument("--layer_norm_affline", default=True)
        parser.add_argument("--step_size", default=100)
        parser.add_argument("--num_split", default=1)
        parser.add_argument("--train_batch_size", default=32)
        parser.add_argument("--val_batch_size", default=32)
        parser.add_argument("--test_batch_size", default=32)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--weight_decay", default=0.0001)
        return parser
