import torch
import numpy as np

from utils.registry import SCALER_REGISTRY
from .base_runner import BaseRunner


class MTGNNRunner(BaseRunner):
    def __init__(self,forward_features,target_features,step_size,num_node,num_split,learning_rate,weight_decay,model,input_len,output_len,dataset_name,**kwargs):
        super().__init__(learning_rate,weight_decay,model,input_len,output_len,dataset_name,forward_features,target_features)
        self.forward_features = forward_features
        self.target_features = target_features
        self.step_size = step_size
        self.num_nodes = num_node
        self.num_split = num_split
        self.perm = None

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        if train:
            future_data, history_data, idx = data
        else:
            future_data, history_data = data
            idx = None

        batch_size, seq_len, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        prediction_data = self.model(
            history_data=history_data, idx=idx, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [
            batch_size, seq_len, num_nodes], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def shared_step(self, batch, batch_idx):
        if batch_idx % self.step_size == 0:
            self.perm = np.random.permutation(range(self.num_nodes))
        num_sub = int(self.num_nodes/self.num_split)
        for j in range(self.num_split):
            if j != self.num_split-1:
                idx = self.perm[j * num_sub:(j + 1) * num_sub]
                raise
            else:
                idx = self.perm[j * num_sub:]
            idx = torch.LongTensor(idx)
            future_data, history_data = batch
            data = future_data[:, :, idx, :], history_data[:, :, idx, :], idx
            prediction, real_value = self(data, batch_idx)
            prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])
            return prediction_rescaled, real_value_rescaled
