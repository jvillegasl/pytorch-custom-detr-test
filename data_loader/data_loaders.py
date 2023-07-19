from pathlib import Path
from torchvision import datasets, transforms

from base import BaseDataLoader
from data_loader.coco import CocoDetection, make_coco_transforms
from utils.misc import collate_fn


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class StarsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = make_coco_transforms('train' if training else 'test')
        self.data_dir = Path(data_dir)
        assert self.data_dir, f'provided COCO path {self.data_dir} does not exist'

        img_folder = self.data_dir / ('train' if training else 'test')
        ann_file = self.data_dir / ('train.json' if training else 'test.json')

        self.dataset = CocoDetection(
            img_folder, ann_file, transforms=trsfm, return_masks=False)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=collate_fn)
