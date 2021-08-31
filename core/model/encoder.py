import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor

from config.base_configs import Configs
from core.model.pos_enc import WordPositionEncoder


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

        num_direction = 2 if self.__C.BIDIRECTIONAL_LSTM else 1

        self.reduce = nn.Linear(in_features=num_direction * __C.ENC_HIDDEN_DIM, out_features=__C.LATENT_DIM)

        self.dropout = nn.Dropout(p=__C.DROPOUT_RATE)

    def forward(self, input_feature: FloatTensor) -> FloatTensor:
        """ Encode bezier features

        Parameters
        ----------
        input_features : FloatTensor
            [b, PAD_SIZE, 10] #10 is the dimension of the extracted bezier features

        Returns
        -------
        FloatTensor
            [b, l, LATENT_DIM]
        """

        feat,_ = self.lstm(input_feature)

        feat_mask = self.make_mask(feat)

        emb = self.pos_enc(feat)

        proj = self.reduce(emb)

        out = self.dropout(proj)

        return out, feat_mask

    def make_mask(self,feature : FloatTensor) -> LongTensor:
        """
        Masking features

        Parameters
        ----------------
        input_features : FloatTensor

        Returns
        ----------------
        LongTensor
        """
        return (torch.sum(
                torch.abs(feature),
                dim = -1
                ) == 0).unsqueeze(1).unsqueeze(2)
