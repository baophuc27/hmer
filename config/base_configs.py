import os
import random
from types import MethodType

import numpy as np
import torch

from config.path_configs import PATH
from core.utils.preprocess import preprocess


class Configs(PATH):
    def __init__(self):
        super(Configs, self).__init__()

        self.GPU = [0]

        self.GPU_STR = "0"
        self.SEED = random.randint(0, 9999999)

        self.VERSION = str(self.SEED)

        # --------------------------------------------------
        # --------------- MODEL PARAM ----------------------
        # --------------------------------------------------

        self.BATCH_SIZE = 32

        self.ENCODER_LSTM_LAYERS = 1

        self.HIDDEN_DIM = 16

        self.BIDIRECTIONAL_LSTM = True

        self.DROPOUT_RATE = 0.3

        self.NUM_WORKERS = 3

        self.PIN_MEMORY = True

        self.MAX_EPOCH = 10

        self.BEAM_SIZE = 3

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith("__") and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def proc(self):
        assert self.RUN_MODE in ["train", "val", "test"]

        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_STR
        torch.backends.cudnn.deterministic = True
        if len(os.listdir(self.DATASET_PATH)) == 0:
            # Padding datasets
            preprocess(self.RAW_PATH, self.DATASET_PATH)

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not isinstance(getattr(self, attr), MethodType):
                print("{ %-17s }->" % attr, getattr(self, attr))

        return ""
