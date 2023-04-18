import argparse

class StemGNN_ShenZhen():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--units", default=156)
        parser.add_argument("--stack_cnt", default=2)
        parser.add_argument("--time_step", default=12)
        parser.add_argument("--multi_layer", default=5)
        parser.add_argument("--horizon", default=12)
        parser.add_argument("--dropout_rate", default=0.5)
        parser.add_argument("--leaky_rate", default=0.2)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        return parser
