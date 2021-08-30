import copy
import os
import time

from pytorch_lightning import Trainer, seed_everything

from config.base_configs import Configs
from core.data.data_module import CROHMEDataModule, CROHMEDataset
from core.data.load_data import CROHMEDataset
from core.model.lit_net import LitNet


class Execution:
    def __init__(self, __C: Configs):
        self.__C = __C

        print("Loading training data ...........")
        self.datamodule = CROHMEDataModule(__C)

        self.datamodule_val = None

        if __C.VAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, "RUN_MODE", "val")

            print("Loading data for validation ..........")
            # self.datamodule_val = CROHMEDataModule(__C_eval)

    def run(self, RUN_MODE):
        if RUN_MODE == "train":
            model = LitNet(self.__C)

            trainer = Trainer()
            trainer.fit(model, datamodule=self.datamodule)
            # print(self.datamodule.dataset[1])
