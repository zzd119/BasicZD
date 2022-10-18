import argparse

class STID_METRLA():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--num_node", default=207)
        parser.add_argument("--input_len", default=12)
        parser.add_argument("--input_dim", default=3)
        parser.add_argument("--embed_dim", default=32)
        parser.add_argument("--output_len", default=12)
        parser.add_argument("--num_layer", default=3)
        parser.add_argument("--if_spatial", default=True)
        parser.add_argument("--node_dim", default=32)
        parser.add_argument("--if_time_in_day", default=True)
        parser.add_argument("--if_day_in_week", default=True)
        parser.add_argument("--temp_dim_tid", default=32)
        parser.add_argument("--temp_dim_diw", default=32)
        parser.add_argument("--train_batch_size", default=32)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0, 1, 2])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.002)
        parser.add_argument("--weight_decay", default=0.0001)
        return parser
