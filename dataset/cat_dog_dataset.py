import os
from torch.utils.data import Dataset

import torch
from PIL import Image
from matplotlib import pyplot as plt
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils import data

from sklearn.model_selection import train_test_split


def plot_images(images, title="", saving_path=""):
    """_summary_

    Args:
        images (_type_): _description_
    """

    plt.figure(figsize=(24, 6))
    grid_imgs = torchvision.utils.make_grid(images)
    np_grid_imgs = grid_imgs.numpy()
    plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))
    plt.title(title)

    if saving_path:
        plt.savefig(os.path.join(saving_path, f"figures/{title}.png"))
    plt.show()


class CatDogDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, imgs, class_to_int, transf=None, root_path=""):

        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.transforms = transf
        self.dir = os.path.abspath(root_path)

    def __getitem__(self, idx):

        image_name = self.imgs[idx]
        img = Image.open(os.path.join(self.dir, image_name))

        label = self.class_to_int(image_name.split(".")[0])
        label = torch.tensor(label, dtype=torch.float32)

        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class CatDogDataloader:
    """_summary_"""

    def __init__(
        self, img_size=224, batch_size=16, dataset_root="", num_workers=2, custom_transform=None
    ) -> None:
        """_summary_"""
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.load_dataset(self.dataset_root, custom_transform)

    def class_to_int(self, x):
        """_summary_

        Args:
            x (str): _description_

        Returns:
            _type_: _description_
        """
        return 0 if "dog" in x.lower() else 1

    def load_dataset(self, dataset_root, custom_transform):
        """_summary_

        Args:
            dataset_root (_type_): _description_
        """
        if not isinstance(custom_transform, transforms.Compose):
            train_transforms = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomCrop(204),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),

                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.means, std=self.stds)
                ]
            )
        else:
            train_transforms = custom_transform

        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.stds)
            ]
        )

        train_path = os.path.join(dataset_root, "train")
        test_path = os.path.join(dataset_root, "validation")

        train_images = os.listdir(os.path.abspath(train_path))
        test_images = os.listdir(os.path.abspath(test_path))

        train_data, valid_data = train_test_split(train_images, test_size=0.1)

        train_dataset = CatDogDataset(
            train_data,
            class_to_int=self.class_to_int,
            transf=train_transforms,
            root_path=os.path.abspath(train_path),
        )
        validation_dataset = CatDogDataset(
            valid_data,
            class_to_int=self.class_to_int,
            transf=test_transforms,
            root_path=os.path.abspath(train_path),
        )
        test_dataset = CatDogDataset(
            test_images,
            class_to_int=self.class_to_int,
            transf=test_transforms,
            root_path=os.path.abspath(test_path),
        )

        self.train_iterator = data.DataLoader(
            dataset=train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

        self.valid_iterator = data.DataLoader(
            dataset=validation_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

        self.test_iterator = data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
