from pathlib import Path
import numpy as np
import pydicom as dcm
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional

def read_dicom(file_path: Path) -> np.ndarray:
    """
    Читает один DICOM-файл и возвращает numpy-массив.
    Args:
        file_path (Path): путь к файлу .dcm
    Returns:
        np.ndarray: изображение (H, W)
    """
    ds = dcm.dcmread(str(file_path))
    img = ds.pixel_array.astype(np.int16)
    return img

def normalize_img(img: np.ndarray, target_median: float = 0.5) -> np.ndarray:
    """
    Нормализует изображение по гамме.
    Используется для исправления сбитой яркости (например, у пневмонии).
    Args:
        img (np.ndarray): входное изображение
        target_median (float): желаемая медиана яркости (0..1)
    Returns:
        np.ndarray: нормализованное изображение
    """
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = ImageOps.autocontrast(pil_img)

    hist = pil_img.histogram()
    cumsum = np.cumsum(hist)
    median_pixel = np.searchsorted(cumsum, cumsum[-1] // 2)
    gamma = np.log(target_median * 255.0) / np.log(max(median_pixel, 1))

    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(gamma)

    return np.array(pil_img)

class DicomDataset(Dataset):
    """
    PyTorch Dataset для DICOM-файлов.
    На входе список файлов и метки классов.

    Args:
        files (List[Path]): список путей к DICOM
        labels (List[int]): список меток
        transform (callable, optional): аугментации
        normalize (bool): включить нормализацию по умолчанию
    """
    def __init__(self, files: List[Path], labels: List[int],
                 transform: Optional[callable] = None, normalize: bool = True):
        self.files = files
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = read_dicom(self.files[idx])

        if self.normalize:
            img = normalize_img(img)

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def load_dicom_dataset(root: Path, class_names: List[str]) -> Tuple[List[Path], List[int]]:
    """
    Сканирует директории и собирает список файлов и меток.
    Каждая папка = отдельный класс.
    Args:
        root (Path): корневая папка (../data)
        class_names (List[str]): список имён папок ( ["norma_anon", "pneumonia_anon", "pneumotorax_anon"])
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

def create_dataloader(dataset: Dataset, batch_size: int = 16,
                      num_workers: int = 4, distributed: bool = False,
                      shuffle: bool = True) -> DataLoader:
    """
    Создаёт DataLoader.
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
