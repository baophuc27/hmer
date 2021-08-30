from random import shuffle

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from config.base_configs import Configs
from core.data.load_data import CROHMEDataset


def collate_fn(batch):
    print(batch)


class CROHMEDataModule(LightningDataModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()
        self.__C = __C
        self.setup()

    def setup(self, stage=None):
        dataset = CROHMEDataset(self.__C)
        print(dataset.data_size)
        self.train_dataset , self.val_dataset = random_split(dataset, ( 1000, dataset.data_size - 1000))

    def train_dataloader(self):
        return DataLoader(dataset = self.train_dataset, 
                          shuffle=True, 
                          batch_size = self.__C.BATCH_SIZE,
                          num_workers=self.__C.NUM_WORKERS, 
                          pin_memory =self.__C.PIN_MEMORY
                          )
    
    def val_dataloader(self):
        return DataLoader(dataset = self.val_dataset, 
                          shuffle=False, 
                          batch_size = self.__C.BATCH_SIZE,
                          num_workers=self.__C.NUM_WORKERS, 
                          pin_memory =self.__C.PIN_MEMORY
                          )

    @property
    def num_classes(self) -> int:
        return self.__C.VOCAB_SIZE
