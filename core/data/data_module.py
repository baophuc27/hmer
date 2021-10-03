import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from config.base_configs import Configs
from core.data.load_data import CROHMEDataset


def collate_tokens(values, pad_idx):
    """Convert a list of 1D tensors into a padded 2d tensor

    Args:
        values ([List[Tensor]]): List of input tensor
        pad_idx ([type]): Index of PADDING token in CROHME dataset
    """
    values = list(map(torch.FloatTensor, values))
    size = max(v.size(0) for v in values)

    res = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: len(v)])

    return res


def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    dst.copy_(src)


def collate_fn(batch):

    feature = [item["feature"] for item in batch]
    label = [item["label"] for item in batch]

    batch_size = batch.__len__()
    feat_size, feat_dim = feature[0].size()

    res = torch.zeros(batch_size, feat_size, feat_dim)

    for i, v in enumerate(feature):
        copy_tensor(v, res[i])

    return res, label


class CROHMEDataModule(LightningDataModule):
    def __init__(self, __C: Configs) -> None:
        super().__init__()
        self.__C = __C
        self.setup()

    def setup(self, stage=None):
        dataset = CROHMEDataset(self.__C)

        self.train_dataset, self.val_dataset = random_split(
            dataset, (1000, dataset.data_size - 1000)
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=False,
            batch_size=self.__C.BATCH_SIZE,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEMORY,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEMORY,
            collate_fn=collate_fn,
        )

    @property
    def num_classes(self) -> int:
        return self.__C.VOCAB_SIZE
