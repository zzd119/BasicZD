import argparse

import torch

from utils import load_adj


class GWNET_METRLA():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        adj_mx, _ = load_adj("datasets/METRLA/output/adj/adj_mx.pkl","doubletransition")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_nodes", default=207)
        parser.add_argument("--supports", default=[torch.tensor(i) for i in adj_mx])
        parser.add_argument("--dropout", default=0.3)
        parser.add_argument("--gcn_bool", default=True)
        parser.add_argument("--addaptadj", default=True)
        parser.add_argument("--aptinit", default=None)
        parser.add_argument("--in_dim", default=2)
        parser.add_argument("--out_dim", default=12)
        parser.add_argument("--residual_channels", default=32)
        parser.add_argument("--dilation_channels", default=32)
        parser.add_argument("--skip_channels", default=256)
        parser.add_argument("--end_channels", default=512)
        parser.add_argument("--kernel_size", default=2)
        parser.add_argument("--blocks", default=4)
        parser.add_argument("--layers", default=2)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--weight_decay", default=0.0001)
        return parser
