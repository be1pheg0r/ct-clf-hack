from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pydicom as dcm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from PIL import Image


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


def save_image(img: np.ndarray, file_path: Path):
    """
    Сохраняет numpy-массив как PNG/JPEG
    Args:
        img (np.ndarray): изображение [0,1] или [0,255]
        file_path (Path): путь для сохранения
    """
    # если в [0,1], то переводим в [0,255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # если изображение одноканальное - делаем "L"
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img)

    pil_img.save(str(file_path))


def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """
    Приводит изображение к диапазону [0,1].
    """
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)


def gamma_correction(img: np.ndarray, gamma: float = 1.0, assume_unit: bool = False) -> np.ndarray:
    """
    Гамма-коррекция.
    gamma < 1 --> светлее, gamma > 1 --> темнее.

    Args:
        img (np.ndarray): входное изображение
        gamma (float): коэффициент гаммы
        assume_unit (bool): если True, предполагаем что вход в [0,1].
                            если False, автоматически нормализуем.

    Returns:
        np.ndarray: изображение [0,1]
    """
    if not assume_unit:
        img = normalize_to_unit(img)

    img = np.clip(img, 0, 1)
    img = np.power(img, 1.0 / gamma)
    return img.astype(np.float32)


def auto_gamma_correction(img: np.ndarray, target: float = 0.5) -> np.ndarray:
    """
    Автоматическая гамма-коррекция по медианной яркости.
    Подгоняет яркость так, чтобы медиана стала близка к target.

    Args:
        img (np.ndarray): входное изображение
        target (float): целевая медианная яркость (по умолчанию 0.5)

    Returns:
        np.ndarray: изображение [0,1]
    """
    img = normalize_to_unit(img)
    median_pixel = np.median(img)
    if median_pixel <= 0: 
        return img
    gamma = np.log(target + 1e-8) / np.log(median_pixel + 1e-8)
    img = np.power(img, gamma)
    return img.astype(np.float32)


class DicomDataset(Dataset):
    """
    Dataset для DICOM-файлов.
    Поддерживает torchvision.transforms (ожидает (H,W,C) numpy).
    """

    def __init__(self,
                 files: List[Path],
                 labels: Optional[List[int]] = None,
                 transform: Optional[callable] = None,
                 normalize: bool = True,
                 gamma: Optional[float] = None,
                 auto_gamma: bool = False,
                 rescale: bool = True,
                 to_rgb: bool = True):
        """
        Args:
            files: список путей к DICOM
            labels: список меток (или None для инференса)
            transform: аугментации torchvision
            normalize: приводить к [0,1]
            gamma: применять фиксированную гамма-коррекцию
            auto_gamma: автоматически подбирать гамму
            rescale: использовать RescaleSlope/Intercept
            to_rgb: конвертировать в RGB (для предобученных моделей)
        """
        self.files = files
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.gamma = gamma
        self.auto_gamma = auto_gamma
        self.rescale = rescale
        self.to_rgb = to_rgb

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # читаем DICOM
        img = read_dicom(self.files[idx], apply_rescale=self.rescale)

        # нормализация
        if self.normalize:
            img = normalize_to_unit(img)

        # гамма
        if self.auto_gamma:
            img = auto_gamma_correction(img)
        elif self.gamma is not None:
            img = gamma_correction(img, self.gamma, assume_unit=self.normalize)

        # grayscale --> RGB
        if self.to_rgb:
            img = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        else:
            img = np.expand_dims(img, axis=-1)        # (H, W, 1)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        return img


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
    DataLoader для PyTorch

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
