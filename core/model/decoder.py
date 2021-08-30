from random import sample
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor
from torch.nn.modules.transformer import TransformerDecoder

from config.base_configs import Configs
from core.data.vocab import CROHMEVocab as Vocab
from core.model.net_utils import Hypothesis, to_tgt_output
from core.model.pos_enc import WordPositionEncoder


class Decoder(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.__C = __C

        self.word_embed = nn.Sequential(nn.Embedding(__C.VOCAB_SIZE, __C.D_MODEL), nn.LayerNorm(__C.D_MODEL))

        self.pos_enc = WordPositionEncoder(__C)

        self.model = self.__class__._build_transformer_decoder(__C)

        self.proj = nn.Linear(__C.D_MODEL, __C.VOCAB_SIZE)

    def forward(self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor) -> FloatTensor:
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
        tgt_pad_mask = tgt == Vocab.PAD_IDX

        tgt = self.pos_enc(self.word_embed(tgt))  # [b , l, d]

        src = rearrange(src, "b t d -> t b d")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt, memory=src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_mask
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    @staticmethod
    def _build_transformer_decoder(__C: Configs) -> nn.TransformerDecoder:

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=__C.D_MODEL, nhead=__C.DEC_NUM_HEAD, dim_feedforward=__C.DEC_FC_DIM, dropout=__C.DROPOUT_RATE
        )
        return TransformerDecoder(decoder_layer, __C.DEC_NUM_LAYER)

    @staticmethod
    def _build_attention_mask(length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full((length, length), fill_value=1, dtype=torch.bool)
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _beam_search(
        self, src: FloatTensor, mask: LongTensor, direction: str, beam_size: int, max_len: int
    ) -> List[Hypothesis]:

        assert direction in {"l2r", "r2l"}
        assert (
            src.size(0) == 1 and mask.size(0) == 1
        ), f"beam search should only have single source, encounter with batch_size: {src.size(0)}"

        if direction == "l2r":
            start_w = Vocab.SOS_IDX
            stop_w = Vocab.EOS_IDX
        else:
            start_w = Vocab.EOS_IDX
            stop_w = Vocab.SOS_IDX

        hypotheses = torch.full((1, max_len + 1), fill_value=Vocab.PAD_IDX, dtype=torch.long, device=self.device)

        hypotheses[:, 0] = start_w

        hyp_score = torch.zeros(1, dtype=torch.float, device=self.device)

        completed_hypotheses: List[Hypothesis] = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)

            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]
            log_p_t = F.log_softmax(decode_outputs, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e= self.__C.)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuous_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // self.__C.VOCAB_SIZE
            hyp_word_ids = top_cand_hyp_pos % self.__C.VOCAB_SIZE

            t += 1
            new_hypotheses = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                hypotheses[prev_hyp_id, t] = hyp_word_id

                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            seq_tensor=hypotheses[prev_hyp_id, 1:t].detach().clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction=direction,
                        )
                    )
                else:
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),
                    score=hyp_scores[0].detach().item(),
                    direction=direction,
                )
            )

        return completed_hypotheses

    def _cross_rate_score(
        self, src: FloatTensor, mask: LongTensor, hypotheses: List[Hypothesis], direction: str,
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask : LongTensor
            [1, l]
        hypotheses : List[Hypothesis]
        direction : str
        """
        assert direction in {"l2r", "r2l"}
        indices = [h.seq for h in hypotheses]
        tgt, output = to_tgt_output(indices, direction, self.device)

        b = tgt.size(0)
        exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
        exp_mask = repeat(mask.squeeze(0), "s -> b s", b=b)

        output_hat = self(exp_src, exp_mask, tgt)

        flat_hat = rearrange(output_hat, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        loss = F.cross_entropy(flat_hat, flat, ignore_index=Vocab.PAD_IDX, reduction="none")

        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, l in enumerate(loss):
            score = -l
            hypotheses[i].score += score

    def beam_search(self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int) -> List[Hypothesis]:
        """run beam search for src features

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        l2r_hypos = self._beam_search(src, mask, "l2r", beam_size, max_len)
        self._cross_rate_score(src, mask, l2r_hypos, direction="r2l")

        r2l_hypos = self._beam_search(src, mask,"r2l",beam_size,max_len)
        self._cross_rate_score(src, mask,r2l_hypos,direction="l2r")
        