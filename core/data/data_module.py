from pytorch_lightning import LightningDataModule
import torch


from config.base_configs import Configs

class DataModule(LightningDataModule):
    def __init__(self,__C : Configs) -> None:
        super().__init__()