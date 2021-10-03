import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor

from config.base_configs import Configs
from core.model.net import Net
from core.model.net_utils import ExpRateRecorder, ce_loss, to_bi_tgt_out


class LitNet(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()
        self.__C = __C
        self.save_hyperparameters()
        self.net = Net(__C)
        self.exprate_recorder = ExpRateRecorder()

    def training_step(self, batch, _):
        feat, label = batch
        
        label, out = to_bi_tgt_out(label, self.device)
        out_hat = self(feat, label)
        loss = ce_loss(out_hat, out)

        self.log(
            "train loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss
    
    def forward(self, features: FloatTensor, label: LongTensor) -> FloatTensor:
        """ Run with bezier features and target label

        Args:
            features (FloatTensor): [b , feat_size , feat_dim] #feat_size = 16, feat_dim = 10
            label (LongTensor): [2b,l]

        Returns:
            FloatTensor: [2b,l,vocab_size]
        """
        return self.net(features, label)

    def validation_step(self,batch, _):
        feat, label = batch
        
        label, out = to_bi_tgt_out(label, self.device)
        out_hat = self(feat, label)
        loss = ce_loss(out_hat, out)

        self.log(
            "val loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        hyps = self.net.beam_search(feat)

        best_hyp = max(hyps, key = lambda h: h.score/ (len(h)**self.__C.BEAM_ALPHA))

        self.exprate_recorder(best_hyp.seq, label[0])
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):

        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.__C.BASE_LR,
            eps=self.__C.OPT_EPS,
            weight_decay=self.__C.WEIGHT_DECAY,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.__C.PATIENCE // self.__C.SCHEDULER_FREQ,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.__C.SCHEDULER_FREQ,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
