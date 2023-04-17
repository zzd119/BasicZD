import argparse

import torch

from utils import load_adj


class DGCRN_PEMS04():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        adj_mx, _ = load_adj("datasets/PEMS04/output/adj/adj_mx.pkl","doubletransition")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--gcn_depth", default=2)
        parser.add_argument("--num_nodes", default=307)
        parser.add_argument("--predefined_A", default=[torch.Tensor(_) for _ in adj_mx])
        parser.add_argument("--dropout", default=0.3)
        parser.add_argument("--subgraph_size", default=20)
        parser.add_argument("--node_dim", default=40)
        parser.add_argument("--middle_dim", default=2)
        parser.add_argument("--seq_length", default=12)
        parser.add_argument("--in_dim", default=2)
        parser.add_argument("--list_weight", default=[0.05, 0.95, 0.95])
        parser.add_argument("--tanhalpha", default=3)
        parser.add_argument("--cl_decay_steps", default=4000)
        parser.add_argument("--rnn_size", default=64)
        parser.add_argument("--hyperGNN_dim", default=16)
        parser.add_argument("--train_batch_size", default=32)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--weight_decay", default=0.0001)
        parser.add_argument("--warm_epochs", default=0)
        parser.add_argument("--cl_epochs", default=6)
        parser.add_argument("--prediction_length", default=12)
        return parser
