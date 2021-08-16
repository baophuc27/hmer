from pytorch_lightning.utilities.cli import LightningCLI
class Execution:
    def __init__(self, __C):
        self.__C = __C

    def run(self, RUN_MODE):
        
        if RUN_MODE == 'train':
            cli = LightningCLI()
