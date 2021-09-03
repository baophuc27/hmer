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

        return {"feature": feature.float(), "label": label}

    def load_data(self):
        max_len = 0
        # Load formula
        formula_dict = {}
        with open(
            os.path.join(self.__C.DATA_PATH[self.__C.RUN_MODE], "caption.txt"), "rb"
        ) as f:
            captions = f.readlines()
        for line in captions:
            tmp = line.decode().strip().split()
            ink = tmp[0]
            formula = tmp[1:]
            formula_dict[ink] = formula
        len_dict = {}
        # Load bezier feature
        for file_name in self.data_list:
            ink_name, post_fix = file_name.split(".")
            if post_fix == "pkl":
                with open(
                    os.path.join(self.__C.DATA_PATH[self.__C.RUN_MODE], file_name), "rb"
                ) as f:
                    data = pickle.load(f)
                    feat = proc_bezier_feat(
                        data["traces"], self.__C.BEZIER_FEAT_PAD_SIZE
                    )
                    self.feature_list.append(feat)

                    label = self.vocab.word2indices(formula_dict[ink_name])
                    # if len(label) > max_len:
                    #     print(" ".join(formula_dict[ink_name]))
                    #     max_len = len(label)
                    if len(label) not in len_dict.keys():
                        len_dict[len(label)] = 1
                    else:
                        len_dict[len(label)] += 1

                    self.label_list.append(label)
                    self.data_size += 1

    def __len__(self):
        return self.data_size
