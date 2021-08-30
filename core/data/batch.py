from dataclasses import dataclass
from torch import FloatTensor, LongTensor
from typing import List

@dataclass
class Batch:
    features : FloatTensor
    symbol : LongTensor
    label_indices : List[int]

