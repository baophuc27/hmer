import numpy as np
import pickle
import time
import torch.utils.data as Data
import functools
import operator
import os
import torch

from config.base_configs import Configs


class CROHMEDataset(Data.Dataset):
    def __init__(self, __C:Configs) -> None:
        super().__init__()


        self.__C = __C
        self.data_list = os.listdir(__C.DATA_PATH[__C.RUN_MODE])
        self.feature_list = []
        self.symbol_list = []
        self.ix_to_label = {}
        self.data_size = 0
        self.load_data()

        print("Total data size: ",self.data_size)
        setattr(__C,'DATA_SIZE',self.data_size)

    def __getitem__(self,idx):
        feature = torch.from_numpy(self.feature_list[idx])
        symbol = torch.from_numpy(self.symbol_list[idx])
        label = self.ix_to_label[idx]

        return {
            "feature": feature,
            "symbol" : symbol,
            "label" : label
        }
    
    def load_data(self):
        for file_name in self.data_list:
            with open(os.path.join(self.__C.DATA_PATH[self.__C.RUN_MODE],file_name),'rb') as f:
                data = pickle.load(f)
                for trace in data['traces']:
                    self.feature_list.append(trace['features'])
                    self.symbol_list.append(trace['label'])
                    self.ix_to_label[self.data_size] = data['annotation']
                    self.data_size += 1
            

    def __len__(self):
        return self.data_size




