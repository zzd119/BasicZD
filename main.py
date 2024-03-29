import argparse
import traceback
import options
import datasets
from util import create_data, runner_mian

parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", default=20)
parser.add_argument("--accelerator", default='gpu')
parser.add_argument("--device", default=[0])
parser.add_argument("--test_predict_point", default=[3,6,12])
parser.add_argument("--cover", default=False)
parser.add_argument("--history_seq_len",default=12)
parser.add_argument("--future_seq_len",default=12)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="The name of the dataset",
    choices=("PEMS04","PEMS07","PEMS08","PEMSBAY","METRLA","ShenZhen","GuangZhou","ShangHai"),
    default="PEMS04"
)
parser.add_argument(
    "--model_name",
    type=str,
    help="The name of the model for spatiotemporal prediction",
    choices=("STID","AGCRN","GWNET","MTGNN","Autoformer","D2STGNN",
             "DCRNN","DGCRN","FEDformer","GTS","Informer","Pyraformer",
             "StemGNN","STGCN","STNorm","TrafficTransformer"),

    default="STID",
)

temp_args, _ = parser.parse_known_args()
parser_dataset = getattr(datasets,temp_args.dataset_name + "_Dataset").add_datasets_specific_arguments(parser)
args_data = parser_dataset.parse_args()
try:
    create_data(args_data)
except:
    traceback.print_exc()

parser_main = getattr(options, temp_args.model_name + "_" + temp_args.dataset_name).add_options_specific_arguments(parser)
args_main = parser_main.parse_args()
try:
    results = runner_mian(args_main)
except:
    traceback.print_exc()

