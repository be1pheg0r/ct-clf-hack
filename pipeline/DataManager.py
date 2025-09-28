from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional, Union
import nibabel as nib
import numpy as np
import pydicom as dicom



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
                 to_rgb: bool = False,
                 to_flat: bool = True):
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
        self.to_flat = to_flat


    def __len__(self) -> int:
        return len(self.files)
    
    @staticmethod
    def normalize_to_unit(img: np.ndarray) -> np.ndarray:
        """
        Приводит изображение к диапазону [0,1].
        """
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        return img.astype(np.float32)

    @staticmethod
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
            img = DicomDataset.normalize_to_unit(img)

        img = np.clip(img, 0, 1)
        img = np.power(img, 1.0 / gamma)
        return img.astype(np.float32)

    @staticmethod
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
        img = DicomDataset.normalize_to_unit(img)
        median_pixel = np.median(img)
        if median_pixel <= 0: 
            return img
        gamma = np.log(target + 1e-8) / np.log(median_pixel + 1e-8)
        img = np.power(img, gamma)
        return img.astype(np.float32)

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
            img = DicomDataset.normalize_to_unit(img)

        # гамма
        if self.auto_gamma:
            img = DicomDataset.auto_gamma_correction(img)
        elif self.gamma is not None:
            img = DicomDataset.gamma_correction(img, self.gamma, assume_unit=self.normalize)

        # grayscale --> RGB
        if self.to_rgb:
            img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        if self.transform:
            img = self.transform(img)
        
        if self.to_flat:
            img = img.flatten() 

        if self.labels is not None:
            return img, label
        return img


class DataManager:
    CLASS_MAP = {
            'norma_anon': 0,
            'pneumonia_anon': 1,
            'pneumotorax_anon': 2,
            'CT-0': 0,
            'CT-1': 3,
            'CT-2': 3,
            'CT-3': 3,
        }
    IDX_TO_CLASS = {
        0: 'norma_anon',
        1: 'pneumonia_anon',
        2: 'pneumotorax_anon',
        0: 'CT-0',
        3: 'CT-1',
        3: 'CT-2',
        3: 'CT-3',
    }

    @staticmethod
    def get_class_name(class_idx: int) -> str:
        return DataManager.IDX_TO_CLASS[class_idx]

    @staticmethod
    def load_images_from_folders(base_dirs):
        images = []
        labels = []

        if isinstance(base_dirs, (str, Path)):
            base_dirs = [base_dirs]

        for base_dir in base_dirs:
            base_dir = Path(base_dir)
            print(f"Processing directory: {base_dir}")

            for class_folder, label in DataManager.CLASS_MAP.items():
                folder_path = base_dir / class_folder
                if not folder_path.is_dir():
                    print(f"Warning: {folder_path} does not exist, skipping.")
                    continue

                for file_path in folder_path.iterdir():
                    if not file_path.is_file():
                        continue

                    if file_path.suffix == '.nii':
                        try:
                            nii_img = nib.load(str(file_path))
                            img_data = nii_img.get_fdata()
                            for slice_idx in range(img_data.shape[-1]):
                                slice_2d = img_data[:, :, slice_idx]
                                if np.max(slice_2d) == 0:
                                    continue
                                slice_min, slice_max = np.min(slice_2d), np.max(slice_2d)
                                if slice_max > slice_min:
                                    slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
                                    slice_normalized = (slice_normalized * 255).astype(np.uint8)
                                    # pil_img = Image.fromarray(slice_normalized).convert("L")
                                    images.append(slice_normalized)
                                    labels.append(label)
                        except Exception as e:
                            print(f"Error with file {file_path}: {e}")
                        continue

                    try:
                        img_array = dicom.dcmread(str(file_path)).pixel_array
                        if img_array.ndim == 3:
                            for slice_idx in range(img_array.shape[0]):
                                slice_2d = img_array[slice_idx]
                                if np.max(slice_2d) == 0:
                                    continue
                                slice_min, slice_max = np.min(slice_2d), np.max(slice_2d)
                                if slice_max > slice_min:
                                    slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
                                    slice_normalized = (slice_normalized * 255).astype(np.uint8)
                                    # pil_img = Image.fromarray(slice_normalized).convert("L")
                                    images.append(slice_normalized)
                                    labels.append(label)
                        else:
                            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
                            img_array = (img_array * 255).astype(np.uint8)
                            # pil_img = Image.fromarray(img_array.squeeze()).convert("L")
                            images.append(img_array)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error with file {file_path}: {e}")

        return images, labels
    
    @staticmethod
    def create_dataset_from_dirs(
        root_dirs: Union[Path, List[Path], str, List[str]],
        transform: Optional[callable] = None,
        normalize: bool = True,
        gamma: Optional[float] = None,
        auto_gamma: bool = False,
        to_rgb: bool = False,
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
        images, labels = DataManager.load_images_from_folders(root_dirs)

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

    @staticmethod
    def create_dataloader(dataset: Dataset,
                        batch_size: int = 16,
                        num_workers: int = 0,
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

    @staticmethod
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
