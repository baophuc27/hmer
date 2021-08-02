import os


class PATH:
    def __init__(self):
        self.RAW_PATH = "./datasets/raw/"
        self.DATASET_PATH = "./datasets/processed/"
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

        self.IMAGE_PATH = {
            "train": self.DATASET_PATH + "",
            "val": self.DATASET_PATH + "",
            "test": self.DATASET_PATH + "",
        }

        self.LABEL_PATH = {"train": self.DATASET_PATH + "", "val": self.DATASET_PATH + ""}

        def check_path(self):
            print("Checking dataset ...")

            for mode in self.IMAGE_PATH:
                if not os.path.exists(self.TARGET_PATH[mode]):
                    print(self.TARGET_PATH[mode] + "NOT EXIST")
                    exit(-1)

            for mode in self.LABEL_PATH:
                if not os.path.exists(self.QUESTION_PATH[mode]):
                    print(self.QUESTION_PATH[mode] + "NOT EXIST")
                    exit(-1)

            print("Finished")
            print("")
