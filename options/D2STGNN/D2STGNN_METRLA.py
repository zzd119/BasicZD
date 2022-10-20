import argparse

import torch

from utils import load_adj


class D2STGNN_METRLA():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        adj_mx, _ = load_adj("datasets/METRLA/output/in{0}_out{1}/adj_mx.pkl".format(temp_args.history_seq_len,temp_args.future_seq_len),"doubletransition")
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_feat", default=1)
        parser.add_argument("--num_hidden", default=32)
        parser.add_argument("--dropout", default=0.1)
        parser.add_argument("--seq_length", default=12)
        parser.add_argument("--k_t", default=3)
        parser.add_argument("--k_s", default=2)
        parser.add_argument("--gap", default=3)
        parser.add_argument("--num_node", default=207)
        parser.add_argument("--adjs", default=[torch.tensor(adj) for adj in adj_mx])
        parser.add_argument("--num_layers", default=5)
        parser.add_argument("--num_modalities", default=2)
        parser.add_argument("--node_hidden", default=10)
        parser.add_argument("--time_emb_dim", default=10)
        parser.add_argument("--train_batch_size", default=32)
        parser.add_argument("--val_batch_size", default=32)
        parser.add_argument("--test_batch_size", default=32)
        parser.add_argument("--forward_features", default=[0,1,2])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.002)
        parser.add_argument("--weight_decay", default=1.0e-5)
        parser.add_argument("--eps", default=1.0e-8)
        return parser
