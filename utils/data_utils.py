from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional, Union
from .utils import iterate_dicom_nii_slices 
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize


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
                 files: List[np.ndarray], # список numpy массивов
                 labels: Optional[List[int]] = None,
                 transform: Optional[callable] = None,
                 normalize: bool = True,
                 gamma: Optional[float] = None,
                 auto_gamma: bool = False,
                 to_rgb: bool = True):
        """
        Args:
            files: список numpy массивов изображений (H, W)
            labels: список меток (или None для инференса)
            transform: аугментации torchvision
            normalize: приводить к [0,1]
            gamma: применять фиксированную гамма-коррекцию
            auto_gamma: автоматически подбирать гамму
            to_rgb: конвертировать в RGB (для предобученных моделей)
        """
        # Теперь DicomDataset ожидает уже загруженные и обработанные срезы (numpy массивы)
        # Или использует iterate_dicom_nii_slices напрямую
        # self.files - это список numpy массивов (H, W)
        self.files = files
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.gamma = gamma
        self.auto_gamma = auto_gamma
        self.to_rgb = to_rgb


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # читаем изображение из списка numpy массивов
        img = self.files[idx] 

        # labels уже должны быть соответствующим образом загружены  или переданы отдельно в __init__
        # или использовать iterate_dicom_nii_slices

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = -1 # или None, если метки неизвестны

        # img - это numpy массив (H, W)
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
            return img, label
        return img


# Вспомогательная функция для создания датасета из директорий
def create_dataset_from_dirs(
    root_dirs: Union[Path, List[Path], str, List[str]],
    class_map: Optional[dict] = None,
    transform: Optional[callable] = None,
    normalize: bool = True,
    gamma: Optional[float] = None,
    auto_gamma: bool = False,
    to_rgb: bool = True,
    show_progress: bool = False
) -> DicomDataset:
    """
    Создает DicomDataset из списка директорий, используя iterate_dicom_nii_slices.

    Args:
        root_dirs: одна или несколько директорий для поиска файлов.
        class_map: словарь соответствия имени папки классу.
        transform: аугментации torchvision.
        normalize: приводить ли к [0,1].
        gamma: применять фиксированную гамма-коррекцию.
        auto_gamma: автоматически подбирать гамму.
        to_rgb: конвертировать в RGB.
        show_progress: показывать прогресс-бар при загрузке.

    Returns:
        DicomDataset: готовый к использованию датасет.
    """
    images = []
    labels = []
    # Используем iterate_dicom_nii_slices для загрузки
    for img_slice, label, file_path in iterate_dicom_nii_slices(
        base_dirs=root_dirs,
        class_map=class_map,
        recursive=True,
        show_progress=show_progress
    ):
        images.append(img_slice)
        labels.append(label)

    # Создаем датасет
    dataset = DicomDataset(
        files=images, # Передаем список numpy массивов
        labels=labels,
        transform=transform,
        normalize=normalize,
        gamma=gamma,
        auto_gamma=auto_gamma,
        to_rgb=to_rgb
    )
    return dataset


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
