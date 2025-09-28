from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import albumentations as A
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Union
from ct_clf_hack.shared.file_utils import iterate_dicom_nii_slices


class MainDataset(Dataset):

    def __init__(self,
                 dpaths: Optional[Union[list[Path], list[str], Path, str]] = None,
                 class_map: Optional[dict] = None,
                 data: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 labels: Optional[Union[list[int], np.ndarray]] = None,
                 transform: Optional[Union[A.Compose, T.Compose, callable]] = None,
                 verbose: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32
                 ) -> None:
        self.images = None
        self.labels = None
        self.transform = transform
        self._verbose = verbose

        if transform is None:
            base_transform = T.Compose([
                T.Resize((512, 512)),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
            self.transform = base_transform

        if dpaths is not None:
            self.images = []
            self.labels = []
            for img, label, _ in iterate_dicom_nii_slices(
                    base_dirs=dpaths,
                    class_map=class_map,
                    recursive=True,
                    verbose=self._verbose
            ):
                self.images.append(img)
                self.labels.append(label)
            np_dtype = np.float32 if dtype == torch.float32 else np.float16
            self.images = torch.tensor(np.array(self.images, dtype=np_dtype), dtype=dtype)
            self.labels = torch.tensor(np.array(self.labels))
            if device is not None:
                self.images = self.images.to(device)
                self.labels = self.labels.to(device)

        elif data is not None and labels is not None:
            if isinstance(data, np.ndarray):
                self.images = torch.tensor(data, dtype=dtype)
            elif isinstance(data, torch.Tensor):
                self.images = data.to(dtype)

            if isinstance(labels, list):
                self.labels = torch.tensor(labels)
            elif isinstance(labels, np.ndarray):
                self.labels = torch.tensor(labels)
            elif isinstance(labels, torch.Tensor):
                self.labels = labels
            if device is not None:
                self.images = self.images.to(device)
                self.labels = self.labels.to(device)

        else:
            raise ValueError(
                "At least one of dpaths or data/labels must be provided"
            )

        assert len(self.images) == len(self.labels), "Images and labels must have the same length"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            if isinstance(self.transform, A.Compose):
                img = self.transform(image=img.numpy())['image']
                img = torch.tensor(img, dtype=torch.float32)
            elif isinstance(self.transform, T.Compose) or callable(self.transform):
                img = img.unsqueeze(0) if img.ndim == 2 else img
                img = self.transform(img)

        return img, label




def create_dataloader(dataset: Dataset,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      device: Optional[torch.device] = None,
                      distributed: bool = False,
                      shuffle: bool = True) -> DataLoader:
    sampler = DistributedSampler(dataset) if distributed and device == torch.device('cuda') else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    return loader


def split_dataset(dataset: Dataset, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset