from pathlib import Path
import numpy as np
import pydicom as dcm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional


def read_dicom(file_path: Path, apply_rescale: bool = True) -> np.ndarray:
    """
    Читает один DICOM-файл и возвращает numpy-массив.
    При необходимости применяет RescaleSlope/RescaleIntercept.

    Args:
        file_path (Path): путь к файлу .dcm
        apply_rescale (bool): применять ли RescaleSlope/Intercept

    Returns:
        np.ndarray: изображение (H, W), float32
    """
    ds = dcm.dcmread(str(file_path))
    img = ds.pixel_array.astype(np.float32)

    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

    return img

def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """
    Приводит изображение к диапазону [0,1].
    Args:
        img (np.ndarray): входное изображение

    Returns:
        np.ndarray: нормализованное изображение [0,1]
    """
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Применяет гамма-коррекцию.
    gamma < 1 -> изображение светлее
    gamma > 1 -> изображение темнее

    Args:
        img (np.ndarray): входное изображение
        gamma (float): коэффициент гаммы
        assume_unit (bool): если True, предполагаем что вход в [0,1].
                            если False, автоматически нормализуем.

    Returns:
        np.ndarray: изображение [0,1]
    """
    img = img.astype(np.float32)

    if not assume_unit:
        img = normalize_to_unit(img)  # нормализуем если данные не в [0,1]

    img = np.clip(img, 0, 1)
    img = np.power(img, gamma)
    return img.astype(np.float32)

class DicomDataset(Dataset):
    """
    PyTorch Dataset для DICOM-файлов.
    На входе список файлов и метки классов.

    Args:
        files (List[Path]): список путей к DICOM
        labels (List[int]): список меток
        transform (callable, optional): аугментации
        normalize (bool): приводить ли к [0,1]
        gamma (float, optional): применить гамма-коррекцию
        rescale (bool): применять RescaleSlope/Intercept
        to_rgb (bool): дублировать канал в RGB 
    """
    def __init__(self,
                 files: List[Path],
                 labels: List[int],
                 transform: Optional[callable] = None,
                 normalize: bool = True,
                 gamma: Optional[float] = None,
                 rescale: bool = True,
                 to_rgb: bool = True): #to_rgb=True --> для моделей с предобученными весами, to_rgb=False --> под ч/б вход.


        self.files = files
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.gamma = gamma
        self.rescale = rescale
        self.to_rgb = to_rgb

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = read_dicom(self.files[idx], apply_rescale=self.rescale)

        # нормализация
        if self.normalize:
            img = normalize_to_unit(img)

        # гамма-коррекция
        if self.gamma is not None:
            img = gamma_correction(img, self.gamma, assume_unit=self.normalize)

        # перевод в тензор
        if self.to_rgb:
            # делаем 3-канальное изображение
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
            img = torch.from_numpy(img).float()
        else:
            img = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)

        label = self.labels[idx]

        # аугментации
        if self.transform:
            img = self.transform(img)

        return img, label

def load_dicom_dataset(root: Path, class_names: List[str]) -> Tuple[List[Path], List[int]]:
    """
    Сканирует директории и собирает список файлов и меток.
    Каждая папка = отдельный класс.

    Args:
        root (Path): корневая папка (../data/uncompressed)
        class_names (List[str]): имена папок (например ["norma_anon", "pneumonia_anon", "pneumotorax_anon"])

    Returns:
        (files, labels): список файлов и меток
    """
    files, labels = [], []
    for label, cls in enumerate(class_names):
        cls_path = root / cls
        dicoms = list(cls_path.glob("*.dcm"))
        files.extend(dicoms)
        labels.extend([label] * len(dicoms))
    return files, labels


def create_dataloader(dataset: Dataset,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      distributed: bool = False,
                      shuffle: bool = True) -> DataLoader:
    """
    Создаёт DataLoader для PyTorch.

    Args:
        dataset (Dataset): PyTorch Dataset
        batch_size (int): размер батча
        num_workers (int): число потоков загрузки
        distributed (bool): включить DistributedSampler (для multi-GPU DDP)
        shuffle (bool): перемешивание (отключается, если distributed=True)

    Returns:
        DataLoader
    """
    sampler = DistributedSampler(dataset) if distributed else None
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
    """
    Делит датасет на train/val.

    Args:
        dataset (Dataset): исходный Dataset
        val_ratio (float): доля валидации

    Returns:
        (train_dataset, val_dataset)
    """
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
