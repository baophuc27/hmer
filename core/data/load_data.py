import functools
import operator
import os
import pickle
import time

import numpy as np
import torch
import torch.utils.data as Data

from config.base_configs import Configs
from core.data.vocab import CROHMEVocab
from core.utils.preprocess import proc_bezier_feat

class CROHMEDataset(Data.Dataset):
    def __init__(self, __C: Configs) -> None:
        super().__init__()

        self.__C = __C
        self.data_list = os.listdir(__C.DATA_PATH[__C.RUN_MODE])
        self.feature_list = []
        self.label_list = []
        self.data_size = 0
        self.vocab = CROHMEVocab()

        self.load_data()

        print("Total data size: ", self.data_size)
        setattr(__C, "DATA_SIZE", self.data_size)
        setattr(__C, "VOCAB_SIZE", self.vocab.__len__())

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.feature_list[idx])
        label = self.label_list[idx]

        return feature.double() , label

    def load_data(self):
        for file_name in self.data_list:
            with open(os.path.join(self.__C.DATA_PATH[self.__C.RUN_MODE], file_name), "rb") as f:
                data = pickle.load(f)
                self.feature_list.append(proc_bezier_feat(data['traces'],self.__C.BEZIER_FEAT_PAD_SIZE))
                self.label_list.append(data['annotation'].replace("$",""))
                self.data_size +=1

    def __len__(self):
        return self.data_size
