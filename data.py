import os
import pytorch_lightning as pl

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


class IntelDataset(Dataset):
    def __init__(
            self,
            target_dir: Path,
            transform: transforms.Compose=None
    ):
        super().__init__()
        self.paths = list(target_dir.glob("*/*.jpg"))
        self.classes, self.class_to_idx = self.find_classes(target_dir)
        self.transform = transform

    def find_classes(self, directory):
        classes = sorted([entry.name for entry in list(os.scandir(directory))])

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure.")
        
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = Image.open(self.paths[index])
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, class_idx
    

class IntelDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            train_transform: transforms.Compose=None,
            test_transform: transforms.Compose=None,
            batch_size: int=32,
            num_workers: int=2
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = IntelDataset(
                target_dir=self.train_dir,
                transform=self.train_transform
            )

            train_size = int(len(self.train_dataset) * 0.8)
            valid_size = len(self.train_dataset) - train_size
            self.train_dataset, self.valid_dataset = random_split(self.train_dataset, [train_size, valid_size])

        if stage == "test" or stage is None:
            self.test_dataset = IntelDataset(
                target_dir=self.test_dir,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )