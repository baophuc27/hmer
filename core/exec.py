import os,copy,time

# from pytorch_lightning.utilities.cli import LightningCLI
from core.data.load_data import CROHMEDataset
from config.base_configs import Configs
from core.data.data_module import CROHMEDataModule, CROHMEDataset
class Execution:
    def __init__(self, __C:Configs):
        self.__C = __C

        print("Loading training data ...........")
        self.datamodule = CROHMEDataModule(__C)
        
        self.datamodule_val = None

        if __C.VAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print("Loading data for validation ..........")
            # self.datamodule_val = CROHMEDataModule(__C_eval)


    def run(self, RUN_MODE):
        print(self.datamodule)
