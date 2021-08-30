from pytorch_lightning import LightningModule

from config.base_configs import Configs
from core.model.net import Net
import torch.optim as optim

from core.model.net_utils import ce_loss

class LitNet(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()
        self.__C = __C
        self.save_hyperparameters()
        self.net = Net(__C)

    def training_step(self, batch  , ix):
        feat, label = batch
        out_hat = self(feat,label)
        loss = ce_loss(out_hat, label)

        return loss
    
    def forward(self, features , label):

        return self.net(features,label)

    def configure_optimizers(self):

        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.__C.BASE_LR,
            eps=1e-6,
            weight_decay=1e-4,
        )

        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.1,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )
        # scheduler = {
        #     "scheduler": reduce_scheduler,
        #     "monitor": "val_ExpRate",
        #     "interval": "epoch",
        #     "frequency": self.trainer.check_val_every_n_epoch,
        #     "strict": True,
        # }
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}