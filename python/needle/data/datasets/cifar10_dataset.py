import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        filenames = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        for filename in filenames:
            with open(os.path.join(base_folder, filename), "rb") as stream:
                batch = pickle.load(stream, encoding="latin1")
            data_key, label_key = (b"data", b"labels") if b"data" in batch else ("data", "labels")
            images.append(batch[data_key])
            labels.append(batch[label_key])

        self.X = np.concatenate(images).reshape(-1, 3, 32, 32).astype(np.float32)
        self.X /= 255.0
        self.y = np.concatenate(labels)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        images = self.X[index]
        if self.transforms and images.size:
            # Transforms use HWC, while the dataset returns CHW.
            transformed = [
                self.apply_transforms(image.transpose(1, 2, 0)).transpose(2, 0, 1)
                for image in images.reshape(-1, 3, 32, 32)
            ]
            images = np.stack(transformed).reshape(images.shape)
        return images, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
