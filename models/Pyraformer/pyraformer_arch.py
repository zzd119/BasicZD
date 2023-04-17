import torch
import torch.nn as nn

from .embed import DataEmbedding, CustomEmbedding
from .layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .layers import EncoderLayer, Decoder, Predictor
from .layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from ..utils import data_transformation_4_xformer


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, model_args):
        super().__init__()

        self.d_model = model_args["d_model"]
        self.window_size = eval(model_args["window_size"])
        self.truncate = model_args["truncate"]
        self.input_size = model_args["input_size"]
        self.decoder = model_args["decoder"]
        self.enc_in = model_args["enc_in"]
        self.truncate = model_args["truncate"]
        self.inner_size = model_args["inner_size"]
        self.d_inner_hid = model_args["d_inner_hid"]
        self.n_head = model_args["n_head"]
        self.d_k = model_args["d_k"]
        self.d_v = model_args["d_v"]
        self.dropout = model_args["dropout"]
        self.n_layer = model_args["n_layer"]
        self.embed_type = model_args["embed_type"]
        self.num_time_features = model_args["num_time_features"]
        self.d_bottleneck = model_args["d_bottleneck"]
        self.use_tvm = model_args["use_tvm"]
        self.CSCM = model_args["CSCM"]

        if self.decoder == 'attention':
            self.mask, self.all_size = get_mask(
                self.input_size, self.window_size, self.inner_size, torch.device("cuda"))
        else:
            self.mask, self.all_size = get_mask(
                self.input_size+1, self.window_size, self.inner_size, torch.device("cuda"))
        self.decoder_type = self.decoder
        if self.decoder == 'FC':
            self.indexes = refer_points(
                self.all_size, self.window_size, torch.device("cuda"))

        if self.use_tvm:
            assert len(set(self.window_size)
                       ) == 1, "Only constant window size is supported."
            padding = 1 if self.decoder == 'FC' else 0
            q_k_mask = get_q_k(self.input_size + padding,
                               self.inner_size, self.window_size[0], torch.device("cuda"))
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(self.d_model, self.d_inner_hid, self.n_head, self.d_k, self.d_v, dropout=self.dropout,
                             normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(self.n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(self.d_model, self.d_inner_hid, self.n_head, self.d_k, self.d_v, dropout=self.dropout,
                             normalize_before=False) for i in range(self.n_layer)
            ])

        if self.embed_type == 'CustomEmbedding':
            # NOTE: Here is different from official code.
            #       We follow the implementation in "Are Transformers Effective for Time Series Forecasting?" (https://arxiv.org/abs/2205.13504).
            # Here is a possible reason:
            #       The custom embedding is not mentioned in the paper,
            #           and it is similar to a cruical technique similar to STID[1] for MTS forecasting, which may cause unfairness.
            #       [1] Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting.
            self.enc_embedding = DataEmbedding(
                self.enc_in, self.d_model, self.num_time_features, self.dropout)
            # self.enc_embedding = CustomEmbedding(self.enc_in, self.d_model, self.covariate_size, self.seq_num, self.dropout)
        else:
            self.enc_embedding = DataEmbedding(
                self.enc_in, self.d_model, self.num_time_features, self.dropout)

        self.conv_layers = eval(self.CSCM)(
            self.d_model, self.window_size, self.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(
                0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Pyraformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, **model_args):
        super().__init__()
        window_size = eval(model_args["window_size"])
        self.predict_step = model_args["predict_step"]
        self.d_model = model_args["d_model"]
        self.input_size = model_args["input_size"]
        self.decoder_type = model_args["decoder"]
        self.channels = model_args["enc_in"]
        self.truncate = model_args["truncate"]

        self.encoder = Encoder(model_args)
        if self.decoder_type == 'attention':
            mask = get_subsequent_mask(
                self.input_size, window_size, self.predict_step, self.truncate)
            self.decoder = Decoder(model_args, mask)
            self.predictor = Predictor(self.d_model, self.channels)
        elif self.decoder_type == 'FC':
            self.predictor = Predictor(
                4 * self.d_model, self.predict_step * self.channels)

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None, dec_self_mask: torch.Tensor = None, dec_enc_mask: torch.Tensor = None) -> torch.Tensor:
        """Feed forward of PyraFormer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in PyraFormer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)
            # NOTE: pre-train is removed.
            # if pretrain:
            #     dec_enc = torch.cat(
            #         [enc_output[:, :self.input_size], dec_enc], dim=1)
            #     pred = self.predictor(dec_enc)
            # else:
            pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(
                enc_output.size(0), self.predict_step, -1)
        return pred.unsqueeze(-1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(
            history_data=history_data, future_data=future_data, start_token_len=0)

        predict_token = torch.zeros(x_enc.size(
            0), 1, x_enc.size(-1), device=x_enc.device)
        x_enc = torch.cat([x_enc, predict_token], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, 0:1, :]], dim=1)

        prediction = self.forward_xformer(
            x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction
