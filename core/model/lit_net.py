from pytorch_lightning import LightningModule

from config.base_configs import Configs
from core.model.net import Net


class LitNet(LightningModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.net = Net(__C)
