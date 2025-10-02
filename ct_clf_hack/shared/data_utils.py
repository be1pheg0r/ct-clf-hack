from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import albumentations as A
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Union, List
from ct_clf_hack.shared.file_utils import iterate_dicom_nii_slices, read_yaml, read_dicom, read_nii

class MainDataset(Dataset):

    def __init__(self,
                 dpaths: Optional[Union[list[Path], list[str], Path, str]] = None,
                 class_map: Optional[dict] = None,
                 data: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 labels: Optional[Union[list[int], np.ndarray]] = None,
                 transform: Optional[Union[A.Compose, T.Compose, callable]] = None,
                 verbose: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 subset: Optional[int] = None,
                 lazy_loading: bool = False
                 ) -> None:
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
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
            self.transform = base_transform

        if dpaths is not None:
            if lazy_loading:
                self._build_lazy_index(dpaths, class_map, subset)
            else:
                self._load_all_data(dpaths, class_map, subset)

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

        elif data is not None and labels is None:
            if isinstance(data, np.ndarray):
                self.images = torch.tensor(data, dtype=dtype)
            elif isinstance(data, torch.Tensor):
                self.images = data.to(dtype)
            if device is not None:
                self.images = self.images.to(device)


    def _build_lazy_index(self, dpaths, class_map, subset):
        """Строит индекс файлов для lazy loading без загрузки изображений"""
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

                # Find DICOM files
                glob_pattern = "**/*"
                dicom_files = [p for p in folder_path.glob(glob_pattern)
                              if p.is_file() and p.suffix.lower() in ['.dcm', '.dicom', '']]

                # Find NII files
                glob_pattern = "**/*.nii"
                nii_files = [p for p in folder_path.glob(glob_pattern) if p.is_file()]

                # Process DICOM files
                for file_path in dicom_files:
                    if subset is not None and count >= subset:
                        return
                    try:
                        import pydicom as dcm
                        ds = dcm.dcmread(str(file_path))
                        img_array = ds.pixel_array

                        if img_array.ndim == 3:
                            # 3D DICOM - add each slice
                            for slice_idx in range(img_array.shape[-1]):
                                if subset is not None and count >= subset:
                                    return
                                self.file_paths.append((file_path, "dcm"))
                                self.slice_indices.append(slice_idx)
                                self.labels.append(label)
                                count += 1
                        else:
                            # 2D DICOM
                            self.file_paths.append((file_path, "dcm"))
                            self.slice_indices.append(None)
                            self.labels.append(label)
                            count += 1
                    except Exception as e:
                        if self._verbose:
                            print(f"Error indexing DICOM file {file_path}: {e}")
                        continue

                # Process NII files
                for file_path in nii_files:
                    if subset is not None and count >= subset:
                        return
                    try:
                        import nibabel as nib
                        nii_img = nib.load(str(file_path))
                        img_data = nii_img.get_fdata().astype(np.float32)

                        # NII files are usually 3D - add each slice
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

    def _load_all_data(self, dpaths, class_map, subset):
        """Загружает все данные в память (старый способ)"""
        self.images = []
        self.labels = []
        for img, label, _ in iterate_dicom_nii_slices(
                base_dirs=dpaths,
                class_map=class_map,
                recursive=True,
                verbose=self._verbose,
                subset=subset
        ):
            self.images.append(img)
            self.labels.append(label)
        max_shape = max(img.shape for img in self.images)
        self.images = [np.pad(img, [(0, max_shape[0] - img.shape[0]),
                                    (0, max_shape[1] - img.shape[1])], mode='constant') if img.shape != max_shape else img
                        for img in self.images]
        np_dtype = np.float32 if self.dtype == torch.float32 else np.float16
        self.images = torch.tensor(np.array(self.images, dtype=np_dtype), dtype=self.dtype)
        self.labels = torch.tensor(np.array(self.labels))
        if self.device is not None:
            self.images = self.images.to(self.device)
            self.labels = self.labels.to(self.device)

    def _load_image_lazy(self, idx: int) -> np.ndarray:
        """Загружает изображение по индексу для lazy loading"""
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
            # Return zero image as fallback
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self) -> int:
        if self.lazy_loading:
            return len(self.labels)
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        if self.lazy_loading:
            # Lazy loading - загружаем изображение при обращении
            img_array = self._load_image_lazy(idx)
            img = torch.tensor(img_array, dtype=self.dtype)
            if self.device is not None:
                img = img.to(self.device)
        else:
            # Обычный режим - изображения уже в памяти
            img = self.images[idx]

        if self.labels:
            label = self.labels[idx]

        if self.transform:
            if isinstance(self.transform, A.Compose):
                img_np = img.numpy() if isinstance(img, torch.Tensor) else img
                transformed = self.transform(image=img_np)
                img = torch.tensor(transformed['image'], dtype=torch.float32)
            elif isinstance(self.transform, T.Compose) or callable(self.transform):
                img = self.transform(img) if img.ndim == 3 else self.transform(img.unsqueeze(0)).squeeze(0)


        if not self.labels:
            return img
        label_int = int(label) if isinstance(label, torch.Tensor) else label
        return img, label_int




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
        pin_memory=(device == torch.device('cuda')),
        sampler=sampler
    )
    return loader


def split_dataset(dataset: Dataset, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset