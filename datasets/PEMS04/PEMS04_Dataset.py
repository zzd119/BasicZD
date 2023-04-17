import argparse


class PEMS04_Dataset():
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_datasets_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        history_seq_len = temp_args.history_seq_len
        future_seq_len = temp_args.future_seq_len
        dataset_name = temp_args.dataset_name
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--output_dir", type=str,
                            default="datasets/{0}/output/in{1}_out{2}".format(dataset_name,history_seq_len, future_seq_len), help="Output directory.")
        parser.add_argument("--adj_dir", type=str,
                            default="datasets/{0}/output/adj".format(dataset_name), help="Spatial directory.")
        parser.add_argument("--data_file_path", type=str,
                            default="datasets/{0}/raw_data/{0}.npz".format(dataset_name), help="Raw traffic readings.")
        parser.add_argument("--graph_file_path", type=str,
                            default="datasets/{0}/raw_data/adj_{0}.pkl".format(dataset_name), help="Raw traffic readings.")
        parser.add_argument("--steps_per_day", type=int,default=288, help="Sequence Length.")
        parser.add_argument("--tod", type=bool, default=True,
                            help="Add feature time_of_day.")
        parser.add_argument("--dow", type=bool, default=True,
                            help="Add feature day_of_week.")
        parser.add_argument("--target_channel", type=list,
                            default=[0], help="Selected channels.")
        parser.add_argument("--train_ratio", type=float,
                            default=0.6, help="Train ratio")
        parser.add_argument("--valid_ratio", type=float,
                            default=0.2, help="Validate ratio.")
        return parser