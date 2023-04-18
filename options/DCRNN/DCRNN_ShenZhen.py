import argparse

import torch

from utils import load_adj


class DCRNN_PEMSBAY():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        adj_mx, _ = load_adj("datasets/ShenZhen/output/adj/adj_mx.pkl", "doubletransition")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--cl_decay_steps", default=2000)
        parser.add_argument("--horizon", default=12)
        parser.add_argument("--input_dim", default=2)
        parser.add_argument("--max_diffusion_step", default=2)
        parser.add_argument("--num_node", default=156)
        parser.add_argument("--num_rnn_layers", default=2)
        parser.add_argument("--output_dim", default=1)
        parser.add_argument("--rnn_units", default=64)
        parser.add_argument("--seq_len", default=12)
        parser.add_argument("--adj_mx", default=[torch.tensor(i).cuda() for i in adj_mx])
        parser.add_argument("--use_curriculum_learning", default=True)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--eps", default=1e-3)
        return parser
