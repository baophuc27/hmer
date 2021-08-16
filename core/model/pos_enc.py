import math
import torch
from pytorch_lightning import LightningModule
from einops import rearrange, repeat

from config.base_configs import Configs

class WordPositionEncoder(LightningModule):
    def __init__(self,__C : Configs) -> None:
        super().__init__()
        
        position_encode = torch.zeros(__C.POS_MAX_LEN,__C.D_MODEL)

        position = torch.arange(0,__C.POS_MAX_LEN, dtype = torch.float)

        dim_t = torch.arange(0,__C.D_MODEL,2,dtype = torch.float)
        div_term = 1.0 / (__C.TEMPERATURE ** (dim_t/ __C.D_MODEL))

        inv_freq = torch.einsum("i, j -> i j", position, div_term)

        position_encode[:, 0::2] = inv_freq.sin()
        position_encode[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", position_encode)
    
    def forward(self, feature : torch.Tensor) -> torch.Tensor:
        """add positional encoding to feature

        Parameters
        ----------
        feature : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, seq_len, _ = feature.size()
        pos_enc = self.position_encode[:seq_len, :]
        feature += pos_enc[None, :, :]
        return feature

    
