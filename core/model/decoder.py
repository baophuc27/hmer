
from random import sample
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange,repeat
from torch import FloatTensor, LongTensor
from torch.nn.modules.transformer import TransformerDecoder

from config.base_configs import Configs
from model.pos_enc import WordPositionEncoder



class Decoder(LightningModule):
    def __init__(self,__C : Configs, vocab_size: int) -> None:
        super().__init__()
        self.word_embed = nn.Sequential(
                nn.Embedding(vocab_size,__C.D_MODEL),
                nn.LayerNorm(__C.D_MODEL)
            )
        
        self.pos_enc = WordPositionEncoder(__C)

        self.model = self.__class__._build_transformer_decoder()

        self.proj = nn.Linear(__C.D_MODEL, vocab_size)
    
    
    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask: LongTensor
            [b, t]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """

        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.pos_enc(self.word_embed(tgt)) # [b , l, d]
        
        src = rearrange(src, "b t d -> t b d")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt = tgt,
            memory = src,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_pad_mask,
            memory_key_padding_mask = src_mask
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)
        
        return out
    
    @staticmethod
    def _build_transformer_decoder(__C : Configs) -> nn.TransformerDecoder:

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = __C.D_MODEL,
            nhead = __C.DEC_NUM_HEAD,
            dim_feedforward = __C.DEC_FC_DIM,
            dropout = __C.DROPOUT_RATE
        )
        return TransformerDecoder(decoder_layer, __C.DEC_NUM_LAYER)

    @staticmethod
    def _build_attention_mask(length):
            # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask