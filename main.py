import argparse
import traceback
import pytorch_lightning as pl
import runner
import options
from pytorch_lightning.utilities import rank_zero_info
from data.spatiotemprol import SpatioTemporalCSVDataModule
from models.AGCRN import AGCRN
from models.GWNET import GWNET
from models.MTGNN import MTGNN
from models.STID import STID


def get_model(args):
    model = None
    if args.model_name == "STID":
        model = STID(**vars(args))
    if args.model_name == "AGCRN":
        model = AGCRN(**vars(args))
    if args.model_name == "GWNET":
        model = GWNET(**vars(args))
    if args.model_name == "MTGNN":
        model = MTGNN(**vars(args))
    return model

def get_task(args, model):
    task = getattr(runner, args.model_name + "Runner")(
        model=model, **vars(args)
    )
    return task

def get_data(args):
    dm = SpatioTemporalCSVDataModule(**vars(args))
    return dm

def main(args):
    rank_zero_info(vars(args))
    dm = get_data(args)
    model = get_model(args)
    task = get_task(args, model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", default=20)
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--device", default=[0])
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset",
        choices=("PEMS04","METRLA"),
        default="PEMS04"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("STID","AGCRN","GWNET","MTGNN"),
        default="STID",
    )
    temp_args, _ = parser.parse_known_args()
    parser = getattr(options, temp_args.model_name + "_" + temp_args.dataset_name).add_options_specific_arguments(parser)
    args = parser.parse_args()

    try:
        results = main(args)

    except:
        traceback.print_exc()

