import torch
from runner import BaseRunner


class GTSRunner(BaseRunner):
    """Runner for DCRNN: add setup_graph and teacher forcing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward_features = kwargs["forward_features"]
        self.target_features = kwargs["target_features"]

    def setup_graph(self, data):
        try:
            self.shared_step(data,0)
        except AttributeError:
            pass

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:

        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True) -> tuple:

        future_data, history_data = data
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = future_data[..., [0]]
        else:
            future_data_4_dec = None
        prediction_data, pred_adj, prior_adj = self.model(history_data,future_data_4_dec,epoch=None,batch_seen=iter_num,train=True)
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value