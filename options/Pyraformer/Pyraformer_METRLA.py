import argparse

import torch


class Pyraformer_METRLA():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_nodes", default=207)
        parser.add_argument("--input_size", default=temp_args.history_seq_len)
        parser.add_argument("--predict_step", default=temp_args.history_seq_len)
        parser.add_argument("--enc_in", default=207)
        parser.add_argument("--dec_in", default=207)
        parser.add_argument("--c_out", default=207)
        parser.add_argument("--d_model", default=512)
        parser.add_argument("--d_inner_hid", default=512)
        parser.add_argument("--d_k", default=128)
        parser.add_argument("--d_v", default=128)
        parser.add_argument("--d_bottleneck", default=128)
        parser.add_argument("--n_head", default=4)
        parser.add_argument("--n_layer", default=4)
        parser.add_argument("--dropout", default=0.05)
        parser.add_argument("--num_time_features", default=2)
        parser.add_argument("--decoder", default="FC")
        parser.add_argument("--window_size", default="[2, 2, 2]")
        parser.add_argument("--inner_size", default=5)
        parser.add_argument("--CSCM", default="Bottleneck_Construct")
        parser.add_argument("--truncate", default=False)
        parser.add_argument("--use_tvm", default=False)
        parser.add_argument("--embed_type", default="DataEmbedding")
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1,2])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.0005)
        parser.add_argument("--weight_decay", default=0.0005)
        return parser
