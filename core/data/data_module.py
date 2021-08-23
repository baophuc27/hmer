from random import shuffle
from pytorch_lightning import LightningDataModule
import torch

from torch.utils.data.dataloader import DataLoader
from config.base_configs import Configs
from core.data.load_data import CROHMEDataset




def collate_fn(batch):
    pass

class CROHMEDataModule(LightningDataModule):
    def __init__(self,__C : Configs) -> None:
        super().__init__()
        self.__C = __C
    
    def setup(self, stage = None):
        self.dataset = CROHMEDataset(self.__C)
    
    def dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle = True,
            num_workers = self.__C.NUM_WORKERS,
            collate_fn = collate_fn
        )