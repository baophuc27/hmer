from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor


from core.model.decoder import Decoder
from core.model.encoder import Encoder
from config.base_configs import Configs


class Net(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.__C = __C
        self.encoder = Encoder(__C)
        self.decoder = Decoder(__C)

    def forward(
        self, features: FloatTensor, features_mask: FloatTensor, symbol: LongTensor, label: LongTensor
    ) -> FloatTensor:
        pass
