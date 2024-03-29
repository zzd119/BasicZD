import os
import pytorch_lightning as pl
import runner
from pytorch_lightning.utilities import rank_zero_info
from data.spatiotemprol import SpatioTemporalCSVDataModule
from datasets.GuangZhou.generate_training_data import GuangZhuo_generate_data
from datasets.METRLA.generate_training_data import METRLA_generate_data
from datasets.PEMS04.generate_training_data import PEMS04_generate_data
from datasets.PEMS07.generate_training_data import PEMS07_generate_data
from datasets.PEMS08.generate_training_data import PEMS08_generate_data
from datasets.PEMSBAY.generate_training_data import PEMSBAY_generate_data
from datasets.ShangHai.generate_training_data import ShangHai_generate_data
from datasets.ShenZhen.generate_training_data import ShenZhen_generate_data
from models.AGCRN import AGCRN
from models.Autoformer import Autoformer
from models.D2STGNN import D2STGNN
from models.DCRNN import DCRNN
from models.DGCRN import DGCRN
from models.FEDformer import FEDformer
from models.GTS import GTS
from models.GWNET import GWNET
from models.Informer import Informer
from models.MTGNN import MTGNN
from models.Pyraformer import Pyraformer
from models.STID import STID
from models.STNorm import STNorm
from models.StemGNN import StemGNN
from models.STGCN import STGCN
from models.TrafficTransformer.trafficTransformer_arch import TrafficTransformer


def create_data(args):
    if os.path.exists(args.output_dir):
        if not args.cover:
            return
    else:
        os.makedirs(args.output_dir)
    if args.dataset_name == "METRLA":
        METRLA_generate_data(args)
    if args.dataset_name == "PEMSBAY":
        PEMSBAY_generate_data(args)
    if args.dataset_name == "PEMS04":
        PEMS04_generate_data(args)
    if args.dataset_name == "PEMS07":
        PEMS07_generate_data(args)
    if args.dataset_name == "PEMS08":
        PEMS08_generate_data(args)
    if args.dataset_name == "ShenZhen":
        ShenZhen_generate_data(args)
    if args.dataset_name == "GuangZhou":
        GuangZhuo_generate_data(args)
    if args.dataset_name == "ShangHai":
        ShangHai_generate_data(args)


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
    if args.model_name == "D2STGNN":
        model = D2STGNN(**vars(args))
    if args.model_name == "DCRNN":
        model = DCRNN(**vars(args))
    if args.model_name == "DGCRN":
        model = DGCRN(**vars(args))
    if args.model_name == "FEDformer":
        model = FEDformer(**vars(args))
    if args.model_name == "GTS":
        model = GTS(**vars(args))
    if args.model_name == "Informer":
        model = Informer(**vars(args))
    if args.model_name == "Pyraformer":
        model = Pyraformer(**vars(args))
    if args.model_name == "StemGNN":
        model = StemGNN(**vars(args))
    if args.model_name == "STGCN":
        model = STGCN(**vars(args))
    if args.model_name == "STNorm":
        model = STNorm(**vars(args))
    if args.model_name == "TrafficTransformer":
        model = TrafficTransformer(**vars(args))
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
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='Val_MAE',
        save_top_k=1,
        mode='min'
    )
    rank_zero_info(vars(args))
    dm = get_data(args)
    model = get_model(args)
    task = get_task(args, model)
    trainer = pl.Trainer.from_argparse_args(args,num_nodes=1,callbacks=[ckpt_callback])
    trainer.fit(task, dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    results = trainer.test(task, dm, ckpt_path='best')
    return results