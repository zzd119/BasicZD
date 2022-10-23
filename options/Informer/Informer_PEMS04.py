import argparse

class Informer_PEMS04():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--num_nodes", default=307)
        parser.add_argument("--enc_in", default=307)
        parser.add_argument("--dec_in", default=307)
        parser.add_argument("--c_out", default=307)
        parser.add_argument("--seq_len", default=temp_args.history_seq_len)
        parser.add_argument("--label_len", default=int(temp_args.history_seq_len/2))
        parser.add_argument("--out_len", default=temp_args.future_seq_len)
        parser.add_argument("--factor", default=5)
        parser.add_argument("--d_model", default=512)
        parser.add_argument("--n_heads", default=8)
        parser.add_argument("--e_layers", default=2)
        parser.add_argument("--d_layers", default=1)
        parser.add_argument("--d_ff", default=2048)
        parser.add_argument("--dropout", default=0.05)
        parser.add_argument("--attn", default="prob")
        parser.add_argument("--activation", default="gelu")
        parser.add_argument("--output_attention", default=False)
        parser.add_argument("--distil", default=True)
        parser.add_argument("--mix", default=True)
        parser.add_argument("--num_time_features", default=2)
        parser.add_argument("--train_batch_size", default=64)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0,1,2])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.0005)
        parser.add_argument("--weight_decay", default=0.0005)
        return parser
