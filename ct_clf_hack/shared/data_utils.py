"""
Модуль для обработки данных медицинских изображений.
Содержит классы датасетов, функции нормализации и преобразования данных,
а также утилиты для работы с PyTorch DataLoader.
"""

import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import albumentations as A
import nibabel as nib
import numpy as np
import pydicom as dicom
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from ct_clf_hack.shared.config_utils import class_map_config
from ct_clf_hack.shared.file_utils import (
    extract_random_slices_from_directory, iterate_dicom_nii_slices, read_dicom,
    read_nii)


def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Преобразует изображение в RGB формат.

    Args:
        img: Входное изображение

    Returns:
        np.ndarray: RGB изображение

    Raises:
        ValueError: При неподдерживаемом формате изображения
    """
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
        return img_rgb
    elif img.ndim == 3 and img.shape[-1] == 1:
        img_rgb = np.concatenate([img, img, img], axis=-1)
        return img_rgb
    elif img.ndim == 3 and img.shape[-1] == 3:
        return img
    else:
        img_2d = np.squeeze(img)
        if img_2d.ndim == 2:
            return np.stack([img_2d, img_2d, img_2d], axis=-1)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Нормализует изображение в диапазон [0, 255] uint8.

    Args:
        img: Входное изображение

    Returns:
        np.ndarray: Нормализованное изображение uint8
    """
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        normalized = (img - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)
    else:
        return np.zeros_like(img, dtype=np.uint8)


def load_images_from_folders(base_dirs: Union[List[Path], List[str], Path, str],
                           class_map: Optional[Dict[str, int]] = None) -> Tuple[List[np.ndarray], List[int]]:
    """
    Загружает изображения из папок с применением карты классов.

    Args:
        base_dirs: Базовые директории для поиска
        class_map: Словарь сопоставления папок и меток классов

    Returns:
        Tuple[List[np.ndarray], List[int]]: Список изображений и меток
    """
    if class_map is None:
        class_map = class_map_config()

    images = []
    labels = []

    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    for img, label, _ in iterate_dicom_nii_slices(
            base_dirs=base_dirs,
            class_map=class_map,
            recursive=True,
            verbose=True
    ):
        normalized = _normalize_image(img)
        images.append(normalized)
        labels.append(label)

    return images, labels


class MainDataset(Dataset):
    """
    Основной класс датасета для медицинских изображений.
    Поддерживает загрузку из файлов или предварительно подготовленных данных.
    """

    def __init__(self,
                 dpaths: Optional[Union[List[Path], List[str], Path, str]] = None,
                 class_map: Optional[Dict[str, int]] = None,
                 data: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 labels: Optional[Union[List[int], np.ndarray]] = None,
                 transform: Optional[Union[A.Compose, T.Compose, callable]] = None,
                 verbose: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 lazy_loading: bool = False,
                 subset: Optional[int] = None
                 ) -> None:
        """
        Инициализация датасета.

        Args:
            dpaths: Пути к данным
            class_map: Карта классов
            data: Предварительно подготовленные данные
            labels: Метки классов
            transform: Трансформации изображений
            verbose: Вывод отладочной информации
            device: Устройство для размещения данных
            dtype: Тип данных тензоров
            lazy_loading: Ленивая загрузка данных
            subset: Ограничение размера данных
        """
        self.images = None
        self.labels = None
        self.transform = transform
        self._verbose = verbose
        self.device = device
        self.dtype = dtype
        self.lazy_loading = lazy_loading

        self.file_paths: List[Tuple[Path, str]] = []
        self.slice_indices: List[Optional[int]] = []

        if transform is None:
            base_transform = T.Compose([
                T.Resize((512, 512)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform = base_transform

        if dpaths is not None:
            if lazy_loading:
                self._build_lazy_index(dpaths, class_map, subset)
            else:
                self._load_all_data(dpaths, class_map, subset)

        elif data is not None and labels is not None:
            self._load_from_arrays(data, labels, device, dtype)

        else:
            raise ValueError("At least one of dpaths or data/labels must be provided")

        assert len(self.images) == len(self.labels), "Images and labels must have the same length"

    def _build_lazy_index(self, dpaths: Union[List[Path], List[str], Path, str],
                         class_map: Optional[Dict[str, int]], subset: Optional[int]) -> None:
        """Строит индекс для ленивой загрузки данных."""
        self.labels = []
        count = 0

        if isinstance(dpaths, (str, Path)):
            dpaths = [dpaths]
        dpaths = [Path(d) for d in dpaths]

        for base_dir in dpaths:
            base_dir = Path(base_dir)
            for class_folder, label in class_map.items():
                folder_path = base_dir / class_folder
                if not folder_path.is_dir():
                    if self._verbose:
                        print(f"Warning: {folder_path} does not exist, skipping.")
                    continue

                dicom_files = [p for p in folder_path.glob("**/*")
                               if p.is_file() and p.suffix.lower() in ['.dcm', '.dicom', '']]
                nii_files = [p for p in folder_path.glob("**/*.nii") if p.is_file()]

                for file_path in dicom_files:
                    if subset is not None and count >= subset:
                        return
                    try:
                        import pydicom as dcm
                        ds = dcm.dcmread(str(file_path))
                        img_array = ds.pixel_array

                        if img_array.ndim == 3:
                            for slice_idx in range(img_array.shape[-1]):
                                if subset is not None and count >= subset:
                                    return
                                self.file_paths.append((file_path, "dcm"))
                                self.slice_indices.append(slice_idx)
                                self.labels.append(label)
                                count += 1
                        else:
                            self.file_paths.append((file_path, "dcm"))
                            self.slice_indices.append(None)
                            self.labels.append(label)
                            count += 1
                    except Exception as e:
                        if self._verbose:
                            print(f"Error indexing DICOM file {file_path}: {e}")
                        continue

                for file_path in nii_files:
                    if subset is not None and count >= subset:
                        return
                    try:
                        import nibabel as nib
                        nii_img = nib.load(str(file_path))
                        img_data = nii_img.get_fdata().astype(np.float32)

                        for slice_idx in range(img_data.shape[-1]):
                            if subset is not None and count >= subset:
                                return
                            self.file_paths.append((file_path, "nii"))
                            self.slice_indices.append(slice_idx)
                            self.labels.append(label)
                            count += 1
                    except Exception as e:
                        if self._verbose:
                            print(f"Error indexing NII file {file_path}: {e}")
                        continue

        self.labels = torch.tensor(np.array(self.labels))
        if self.device is not None:
            self.labels = self.labels.to(self.device)

    def _load_all_data(self, dpaths: Union[List[Path], List[str], Path, str],
                      class_map: Optional[Dict[str, int]], subset: Optional[int]) -> None:
        """Загружает все данные в память."""
        self.images = []
        self.labels = []
        for img, label, _ in iterate_dicom_nii_slices(
                base_dirs=dpaths,
                class_map=class_map,
                recursive=True,
                verbose=self._verbose,
                subset=subset
        ):
            img_rgb = convert_to_rgb(img)
            self.images.append(img_rgb)
            self.labels.append(label)

        normalized_images = []
        for img in self.images:
            img_norm = img.astype(np.float32)
            for c in range(img_norm.shape[-1]):
                channel = img_norm[:, :, c]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    img_norm[:, :, c] = (channel - min_val) / (max_val - min_val)
                else:
                    img_norm[:, :, c] = 0
            normalized_images.append(img_norm)

        np_dtype = np.float32 if self.dtype == torch.float32 else np.float16
        self.images = torch.tensor(np.array(normalized_images, dtype=np_dtype), dtype=self.dtype)
        self.images = self.images.permute(0, 3, 1, 2)
        self.labels = torch.tensor(np.array(self.labels))
        if self.device is not None:
            self.images = self.images.to(self.device)
            self.labels = self.labels.to(self.device)

    def _load_from_arrays(self, data: Union[np.ndarray, torch.Tensor],
                         labels: Union[List[int], np.ndarray],
                         device: Optional[torch.device], dtype: torch.dtype) -> None:
        """Загружает данные из готовых массивов."""
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                data_rgb = np.stack([data, data, data], axis=-1)
                data_rgb = np.transpose(data_rgb, (0, 3, 1, 2))
            elif data.ndim == 4 and data.shape[-1] == 1:
                data_rgb = np.concatenate([data, data, data], axis=-1)
                data_rgb = np.transpose(data_rgb, (0, 3, 1, 2))
            elif data.ndim == 4 and data.shape[1] == 1:
                data_rgb = np.concatenate([data, data, data], axis=1)
            else:
                data_rgb = data
            self.images = torch.tensor(data_rgb, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            if data.ndim == 3:
                data_rgb = torch.stack([data, data, data], dim=-1)
                data_rgb = data_rgb.permute(0, 3, 1, 2)
            elif data.ndim == 4 and data.shape[-1] == 1:
                data_rgb = torch.cat([data, data, data], dim=-1)
                data_rgb = data_rgb.permute(0, 3, 1, 2)
            elif data.ndim == 4 and data.shape[1] == 1:
                data_rgb = torch.cat([data, data, data], dim=1)
            else:
                data_rgb = data
            self.images = data_rgb.to(dtype)

        if isinstance(labels, list):
            self.labels = torch.tensor(labels)
        elif isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels)
        elif isinstance(labels, torch.Tensor):
            self.labels = labels
        if device is not None:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)

    def _load_image_lazy(self, idx: int) -> np.ndarray:
        """
        Ленивая загрузка изображения по индексу.

        Args:
            idx: Индекс изображения

        Returns:
            np.ndarray: Загруженное изображение
        """
        file_path, file_type = self.file_paths[idx]
        slice_idx = self.slice_indices[idx]

        try:
            if file_type == "dcm":
                img_array = read_dicom(file_path)
                if img_array.ndim == 3 and slice_idx is not None:
                    img_array = img_array[:, :, slice_idx]
                else:
                    img_array = img_array.squeeze()
            elif file_type == "nii":
                img_3d = read_nii(file_path)
                if slice_idx is not None:
                    img_array = img_3d[:, :, slice_idx]
                else:
                    img_array = img_3d.squeeze()
            else:
                raise ValueError(f"Unknown file type: {file_type}")

            return img_array.astype(np.float32)

        except Exception as e:
            if self._verbose:
                print(f"Error loading image {file_path}, slice {slice_idx}: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        if self.lazy_loading:
            return len(self.labels)
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Tuple[torch.Tensor, int]: Изображение и метка класса
        """
        if self.lazy_loading:
            img_array = self._load_image_lazy(idx)
            img = torch.tensor(img_array, dtype=self.dtype)
            if self.device is not None:
                img = img.to(self.device)
        else:
            img = self.images[idx]

        label = self.labels[idx]

        if self.transform:
            if isinstance(self.transform, A.Compose):
                img_hwc = img.permute(1, 2, 0).numpy()
                img_transformed = self.transform(image=img_hwc)['image']
                img = torch.tensor(img_transformed, dtype=torch.float32)
                if img.ndim == 3:
                    img = img.permute(2, 0, 1)
            elif isinstance(self.transform, T.Compose) or callable(self.transform):
                img = self.transform(img)

        label_int = int(label) if isinstance(label, torch.Tensor) else label
        return img, label_int


class CTDataset(Dataset):
    """Простой датасет для CT изображений."""

    def __init__(self, images: List[np.ndarray], labels: Optional[List[int]] = None,
                 transform: Optional[callable] = None) -> None:
        """
        Инициализация CT датасета.

        Args:
            images: Список изображений
            labels: Список меток (опционально)
            transform: Трансформации изображений
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """
        Получает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, int]]: Изображение или изображение с меткой
        """
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


def create_dataloader(dataset: Dataset,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      device: Optional[torch.device] = None,
                      distributed: bool = False,
                      shuffle: bool = True) -> DataLoader:
    """
    Создает DataLoader для датасета.

    Args:
        dataset: Датасет
        batch_size: Размер батча
        num_workers: Количество воркеров
        device: Устройство
        distributed: Распределенное обучение
        shuffle: Перемешивание данных

    Returns:
        DataLoader: Загрузчик данных
    """
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
    """
    Разделяет датасет на тренировочную и валидационную части.

    Args:
        dataset: Исходный датасет
        val_ratio: Доля валидационных данных

    Returns:
        Tuple[Dataset, Dataset]: Тренировочный и валидационный датасеты
    """
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def process_zip_dicom_for_backend(zip_path: Union[str, Path], max_slices: int = 10,
                                delete: bool = True) -> Tuple[List[np.ndarray], str]:
    """
    Обрабатывает ZIP архив с DICOM файлами для бэкенда.

    Args:
        zip_path: Путь к ZIP файлу
        max_slices: Максимальное количество срезов
        delete: Удалять ли ZIP файл после обработки

    Returns:
        Tuple[List[np.ndarray], str]: Список изображений и статус обработки
    """
    zip_path = Path(zip_path)
    temp_dir = None

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="dicom_processing_"))

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images = extract_random_slices_from_directory(temp_dir, max_slices)
        normalized_images = [_normalize_image(img) for img in images]

        if delete and zip_path.exists():
            zip_path.unlink()

        status = f"Successfully processed {len(normalized_images)} slices"
        return normalized_images, status

    except Exception as e:
        status = f"Error processing zip archive: {str(e)}"
        return [], status

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_uploaded_dicom_zip(zip_bytes: bytes, max_slices: int = 10) -> Tuple[List[np.ndarray], str]:
    """
    Обрабатывает загруженный ZIP архив с DICOM файлами.

    Args:
        zip_bytes: Байты ZIP архива
        max_slices: Максимальное количество срезов

    Returns:
        Tuple[List[np.ndarray], str]: Список изображений и статус обработки
    """
    temp_zip = None
    temp_dir = None

    try:
        temp_zip = Path(tempfile.mktemp(suffix=".zip"))
        with open(temp_zip, 'wb') as f:
            f.write(zip_bytes)

        temp_dir = Path(tempfile.mkdtemp(prefix="dicom_upload_"))

        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images = extract_random_slices_from_directory(temp_dir, max_slices)
        normalized_images = [_normalize_image(img) for img in images]

        status = f"Successfully processed {len(normalized_images)} slices from upload"
        return normalized_images, status

    except Exception as e:
        status = f"Error processing uploaded zip: {str(e)}"
        return [], status

    finally:
        if temp_zip and temp_zip.exists():
            temp_zip.unlink()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_images_for_model(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Подготавливает изображения для модели, преобразуя их в RGB формат.

    Args:
        images: Список входных изображений

    Returns:
        List[np.ndarray]: Список RGB изображений
    """
    rgb_images = []
    for img in images:
        img_rgb = convert_to_rgb(img)
        rgb_images.append(img_rgb)

    return rgb_images
