import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor


from core.model.pos_enc import WordPositionEncoder
from config.base_configs import Configs


class Encoder(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.__C = __C

        self.lstm = nn.LSTM(
            input_size=__C.EMBED_SIZE,
            hidden_size=__C.ENC_HIDDEN_DIM,
            num_layers=__C.ENCODER_LSTM_LAYERS,
            batch_first=True,
            bidirectional=__C.BIDIRECTIONAL_LSTM,
        )

        self.pos_enc = WordPositionEncoder(__C)

        self.reduce = nn.Linear(in_features=__C.ENC_HIDDEN_DIM, out_features=__C.HIDDEN_DIM)

        self.dropout = nn.Dropout(p=__C.DROPOUT_RATE)

    def forward(self, input_feature: FloatTensor) -> FloatTensor:
        """ Encode bezier features

        Parameters
        ----------
        input_features : FloatTensor
            [b, t, d]

        Returns
        -------
        FloatTensor
            [b, l, HIDDEN_DIM]
        """

        feat = self.lstm(input_feature)

        emb = self.pos_enc(feat)

        proj = self.reduce(emb)

        out = self.dropout(proj)

        return out
