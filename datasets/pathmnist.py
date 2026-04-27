import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from core.data import BaseDataset, subsample_data, to_one_hot


class PathMnist(BaseDataset):
    """
    MedMNIST PathMNIST (9-class, 28x28 RGB). Requires: pip install medmnist
    """

    def __init__(self, cache_folder: str, config: dict, pool_rng, encoded: bool, data_file="pathmnist_al.pt"):
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)

    @staticmethod
    def _stack_split(root: str, split: str):
        try:
            from medmnist import PathMNIST
        except ImportError as e:
            raise ImportError("PathMNIST requires the medmnist package: pip install medmnist") from e

        ds = PathMNIST(split=split, download=True, root=root)
        to_tensor = transforms.ToTensor()

        def collate(batch):
            xs = torch.stack([to_tensor(b[0]) for b in batch])
            ys = []
            for b in batch:
                lab = b[1]
                if torch.is_tensor(lab):
                    ys.append(int(lab.view(-1)[0].item()))
                else:
                    ys.append(int(np.asarray(lab).reshape(-1)[0]))
            return xs, torch.tensor(ys, dtype=torch.long)

        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate)
        parts_x, parts_y = [], []
        for xb, yb in loader:
            parts_x.append(xb)
            parts_y.append(yb)
        x = torch.cat(parts_x, dim=0)
        y = torch.cat(parts_y, dim=0)
        y_oh = to_one_hot(y.numpy().astype(int))
        return x, y_oh

    def _download_data(self, target_to_one_hot=True, test_data_fraction=0.5):
        x_train, self.y_train = self._stack_split(self.cache_folder, "train")
        x_test, y_test = self._stack_split(self.cache_folder, "test")
        x_test, self.y_test = subsample_data(x_test, y_test, test_data_fraction, self.pool_rng)
        self.x_train = x_train
        self.x_test = x_test
        self._convert_data_to_tensors()
        normalizer = transforms.Compose(
            [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        self.x_train = normalizer(self.x_train)
        self.x_test = normalizer(self.x_test)
        print("Download successful")

    def load_pretext_data(self) -> tuple[Dataset, Dataset]:
        try:
            from medmnist import PathMNIST
        except ImportError as e:
            raise ImportError("PathMNIST requires the medmnist package: pip install medmnist") from e

        train_dataset = PathMNIST(split="train", download=True, root=self.cache_folder)
        val_dataset = PathMNIST(split="val", download=True, root=self.cache_folder)
        return train_dataset, val_dataset

    def get_pretext_transforms(self, config: dict) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=28),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def get_pretext_validation_transforms(self, config: dict) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def get_meta_data(self) -> str:
        s = super().get_meta_data() + "\n"
        s += "Source: MedMNIST PathMNIST\n" "Classifier: ResNet18 (3x28x28)"
        return s
