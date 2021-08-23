import os


class PATH:
    def __init__(self):
        self.DATASET_PATH = "./datasets/"
        self.init_path()

    def init_path(self):
        self.PRED_PATH = "./results/pred/"
        self.LOG_PATH = "./results/log/"
        self.CKPTS_PATH = "./ckpts/"

        if "pred" not in os.listdir("./results"):
            os.mkdir("./results/pred")

        if "log" not in os.listdir("./results"):
            os.mkdir("./results/log")

        if "ckpts" not in os.listdir("./"):
            os.mkdir("./ckpts")

        if "processed" not in os.listdir("./datasets"):
            os.mkdir(("./datasets/processed"))

        self.DATA_PATH = {
            "train": self.DATASET_PATH + "TEST2016_INKML_GT",
            "val": self.DATASET_PATH + "",
            "test": self.DATASET_PATH + "",
        }

        self.check_path()

    def check_path(self):
        print("Checking dataset ...")

        for mode in self.DATA_PATH:
            if not os.path.exists(self.DATA_PATH[mode]):
                print(self.DATA_PATH[mode] + "NOT EXIST")
                exit(-1)

        print("Finished")
        print("")
