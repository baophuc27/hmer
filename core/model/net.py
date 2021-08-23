from random import sample
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import FloatTensor, LongTensor

from config.base_configs import Configs
from model.encoder import Encoder
from model.decoder import Decoder

class Net(LightningModule):
    def __init__(self,__C : Configs) -> None:
        super().__init__()

        self.__C = __C
        self.encoder = Encoder(__C)
        self.decoder = Decoder(__C)
    
    def forward(self,features, symbol, label):
        pass