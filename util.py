import os
import pytorch_lightning as pl
import runner
from pytorch_lightning.utilities import rank_zero_info
from data.spatiotemprol import SpatioTemporalCSVDataModule
from datasets.METRLA.generate_training_data import METRLA_generate_data
from datasets.PEMS04.generate_training_data import PEMS04_generate_data
from models.AGCRN import AGCRN
from models.Autoformer import Autoformer
from models.GWNET import GWNET
from models.MTGNN import MTGNN
from models.STID import STID

def create_data(args,cover=False):
    if os.path.exists(args.output_dir):
        if not cover:
            return
    else:
        os.makedirs(args.output_dir)
    if args.dataset_name == "METRLA":
        METRLA_generate_data(args)
    if args.dataset_name == "PEMS04":
        PEMS04_generate_data(args)

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
    if args.model_name == "Autoformer":
        model = Autoformer(**vars(args))
    return model

def get_task(args, model):
    task = getattr(runner, args.model_name + "Runner")(
        model=model, **vars(args)
    )
    return task

def get_data(args):
    dm = SpatioTemporalCSVDataModule(**vars(args))
    return dm

def runner_mian(args):
    rank_zero_info(vars(args))
    dm = get_data(args)
    model = get_model(args)
    task = get_task(args, model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results