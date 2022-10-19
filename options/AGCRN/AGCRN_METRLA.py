import argparse

class AGCRN_METRLA():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_node", default=207)
        parser.add_argument("--input_dim", default=2)
        parser.add_argument("--output_dim", default=1)
        parser.add_argument("--rnn_units", default=64)
        parser.add_argument("--horizon", default=12)
        parser.add_argument("--num_layers", default=2)
        parser.add_argument("--default_graph", default=True)
        parser.add_argument("--embed_dim", default=10)
        parser.add_argument("--cheb_k", default=2)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.003)
        parser.add_argument("--weight_decay", default=0)
        return parser
