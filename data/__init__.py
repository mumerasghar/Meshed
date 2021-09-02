from .field import RawField, Merge, ImageDetectionsField, TextField
from .dataset import COCO
from torch.utils.data import DataLoader as TorchDataLoader
from .dataset import Flicker8k


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, **kwargs)
