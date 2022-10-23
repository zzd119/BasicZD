import argparse

from utils import load_pkl


class GTS_PEMS04():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        node_feats_full = load_pkl("datasets/PEMS04/output/in{0}_out{1}/data_in{0}_out{1}.pkl".format(temp_args.history_seq_len,temp_args.future_seq_len))["processed_data"][..., 0]
        train_index_list = load_pkl("datasets/PEMS04/output/in{0}_out{1}/index_in{0}_out{1}.pkl".format(temp_args.history_seq_len,temp_args.future_seq_len))["train"]
        node_feats = node_feats_full[:train_index_list[-1][-1], ...]
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--cl_decay_steps", default=2000)
        parser.add_argument("--filter_type", default="dual_random_walk")
        parser.add_argument("--horizon", default=12)
        parser.add_argument("--input_dim", default=2)
        parser.add_argument("--l1_decay", default=0)
        parser.add_argument("--max_diffusion_step", default=3)
        parser.add_argument("--num_nodes", default=307)
        parser.add_argument("--num_rnn_layers", default=1)
        parser.add_argument("--output_dim", default=1)
        parser.add_argument("--rnn_units", default=64)
        parser.add_argument("--seq_len", default=12)
        parser.add_argument("--use_curriculum_learning", default=True)
        parser.add_argument("--dim_fc", default=162976)
        parser.add_argument("--node_feats", default=node_feats)
        parser.add_argument("--temp", default=0.5)
        parser.add_argument("--k", default=30)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--eps", default=1e-3)
        return parser
