import os
import random
from types import MethodType

import numpy as np
import torch

from config.path_configs import PATH


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

        self.BATCH_SIZE = 4

        self.ENCODER_LSTM_LAYERS = 1

        self.LATENT_DIM = 64

        self.BIDIRECTIONAL_LSTM = True

        self.DROPOUT_RATE = 0.3

        self.NUM_WORKERS = 3

        self.PIN_MEMORY = True

        self.MAX_EPOCH = 10

        self.BEAM_SIZE = 3

        self.VAL_EVERY_EPOCH = True

        self.K_FOLD = 5

        self.BEZIER_FEAT_PAD_SIZE = 16

        # ------------------------------------------------
        # ----------------- ENCODER ----------------------
        # ------------------------------------------------
        self.EMBED_SIZE = 10

        self.ENC_HIDDEN_DIM = 128

        self.ENCODER_LSTM_LAYERS = 4

        self.BIDIRECTIONAL_LSTM = True

        # ------------------------------------------------
        # ----------------- DECODER ----------------------
        # ------------------------------------------------
        # Decoder num head
        self.DEC_NUM_HEAD = 4

        # Decoder feedfoward dimension
        self.DEC_FC_DIM = 128

        # Number of decoder layer
        self.DEC_NUM_LAYER = 2

        # -------------------------------------------------
        # -------- POSITIONAL ENCODING --------------------
        # -------------------------------------------------
        # Dimension model ---- TODO: verify its meaning
        self.D_MODEL = 128

        self.POS_MAX_LEN = 500

        self.TEMPERATURE = 10000.0

        # ------------------------------------------------
        # -------------- OPTIMIZER -----------------------
        # ------------------------------------------------

        self.OPT_EPS = 1e-6

        self.WEIGHT_DECAY = 1e-4

        self.BASE_LR = 1.0

        self.SCHEDULER_FREQ = 2

        self.PATIENCE = 20

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith("__") and not isinstance(
                getattr(args, arg), MethodType
            ):
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

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not isinstance(
                getattr(self, attr), MethodType
            ):
                print("{ %-17s }->" % attr, getattr(self, attr))

        return ""
