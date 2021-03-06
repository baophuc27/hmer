from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor

from config.base_configs import Configs
from core.model.decoder import Decoder
from core.model.encoder import Encoder


class Net(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.__C = __C
        self.encoder = Encoder(__C)
        self.decoder = Decoder(__C)

    def forward(self, in_feature: FloatTensor, label: LongTensor) -> FloatTensor:
        """ Feed bezier feature and bi-tgt
        
        Parameters
        ----------
        feature : FloatTensor
                [b, 10, n]
        tgt : LongTensor

        Returns
        ----------
        FloatTensor
            [2b , l, vocab_size]
        """

        feature, feature_mask = self.encoder(in_feature)
        feature = torch.cat((feature, feature), dim=0)
        feature_mask = torch.cat((feature_mask, feature_mask), dim=0)

        out = self.decoder(feature, feature_mask, label)

        return out

    def beam_search(self,in_feature):
        feature , feature_mask = self.encoder(in_feature)
        return self.decoder.beam_search(feature,feature_mask,self.__C.BEAM_SIZE, self.__C.BEAM_MAX_LEN)