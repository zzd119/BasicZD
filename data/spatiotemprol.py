import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesForecastingDataset


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(self,dataset_name,input_len,output_len,train_batch_size,val_batch_size,test_batch_size,**kwargs) -> None:
        super(SpatioTemporalCSVDataModule,self).__init__()
        self.dataset_name = dataset_name
        self.input_len = input_len
        self.output_len = output_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage: str = None):
        data_file_path = "./datasets/{0}/output/in{1}_out{2}/data_in{1}_out{2}.pkl".format(self.dataset_name,self.input_len,self.output_len)
        index_file_path = "./datasets/{0}/output/in{1}_out{2}/index_in{1}_out{2}.pkl".format(self.dataset_name,self.input_len,self.output_len)
        self.train_dataset = TimeSeriesForecastingDataset(data_file_path, index_file_path, "train")
        self.val_dataset = TimeSeriesForecastingDataset(data_file_path, index_file_path, "valid")
        self.test_dataset = TimeSeriesForecastingDataset(data_file_path,index_file_path,"test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.train_batch_size,num_workers=0,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.val_batch_size,num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.test_batch_size,num_workers=0)


if __name__ == "__main__":
    sp = SpatioTemporalCSVDataModule("PEMS04",12,12,32,32,32)
    sp.setup()
    x,y = next(iter(sp.train_dataset))
    print(x.shape)