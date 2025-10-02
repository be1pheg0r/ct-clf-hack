"""
Модуль для работы с файлами DICOM и NII.
Содержит функции для чтения, обработки и извлечения данных из медицинских изображений.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import random

import nibabel as nib
import numpy as np
import pydicom as dcm
import yaml
from PIL import Image
from tqdm import tqdm


def read_yaml(file_path: Path) -> dict:
    """
    Читает YAML файл и возвращает данные в виде словаря.

    Args:
        file_path: Путь к YAML файлу

    Returns:
        dict: Данные из YAML файла
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def read_dicom(file_path: Path, apply_rescale: bool = True) -> np.ndarray:
    """
    Читает DICOM файл и возвращает изображение в виде numpy массива.

    Args:
        file_path: Путь к DICOM файлу
        apply_rescale: Применять ли rescale slope и intercept

    Returns:
        np.ndarray: Изображение в виде массива float32
    """
    ds = dcm.dcmread(str(file_path))
    img = ds.pixel_array.astype(np.float32)

    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

    return img


def read_nii(file_path: Path) -> np.ndarray:
    """
    Читает NII файл и возвращает изображение в виде numpy массива.

    Args:
        file_path: Путь к NII файлу

    Returns:
        np.ndarray: Изображение в виде массива float32
    """
    nii_img = nib.load(str(file_path))
    img_data = nii_img.get_fdata().astype(np.float32)
    return img_data


def save_image(img: np.ndarray, file_path: Path) -> None:
    """
    Сохраняет изображение в файл.

    Args:
        img: Изображение в виде numpy массива
        file_path: Путь для сохранения изображения
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img)

    pil_img.save(str(file_path))


def is_probable_dicom_file(file_path: Path) -> bool:
    """
    Проверяет, является ли файл DICOM файлом.

    Args:
        file_path: Путь к файлу для проверки

    Returns:
        bool: True если файл вероятно является DICOM файлом
    """
    try:
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True

        with open(file_path, 'rb') as f:
            header = f.read(132)
            if len(header) > 132 and header[128:132] == b"DICM":
                return True

        try:
            ds = dcm.dcmread(str(file_path), stop_before_pixels=True, force=True)
            if hasattr(ds, "SOPClassUID") or hasattr(ds, "StudyInstanceUID"):
                return True
        except Exception:
            return False

    except Exception:
        return False

    return False


def find_dicom_files(directory: Path) -> List[Path]:
    """
    Находит все DICOM файлы в директории рекурсивно.

    Args:
        directory: Директория для поиска

    Returns:
        List[Path]: Список путей к найденным DICOM файлам
    """
    dicom_files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            if is_probable_dicom_file(file_path):
                dicom_files.append(file_path)

    return dicom_files


def extract_slices_from_dicom(file_path: Path) -> List[np.ndarray]:
    """
    Извлекает срезы из DICOM файла.

    Args:
        file_path: Путь к DICOM файлу

    Returns:
        List[np.ndarray]: Список извлеченных срезов
    """
    slices = []
    try:
        ds = dcm.dcmread(str(file_path))
        img_array = ds.pixel_array.astype(np.float32)

        if img_array.ndim == 3:
            for i in range(img_array.shape[-1]):
                slice_2d = img_array[:, :, i]
                if np.max(slice_2d) > 0:
                    slices.append(slice_2d)
        elif img_array.ndim == 2:
            if np.max(img_array) > 0:
                slices.append(img_array)
    except Exception as e:
        print(f"Error processing DICOM file {file_path}: {e}")

    return slices


def extract_slices_from_nii(file_path: Path) -> List[np.ndarray]:
    """
    Извлекает срезы из NII файла.

    Args:
        file_path: Путь к NII файлу

    Returns:
        List[np.ndarray]: Список извлеченных срезов
    """
    slices = []
    try:
        nii_img = nib.load(str(file_path))
        img_data = nii_img.get_fdata()

        for i in range(img_data.shape[-1]):
            slice_2d = img_data[:, :, i]
            if np.max(slice_2d) > 0:
                slices.append(slice_2d)
    except Exception as e:
        print(f"Error processing NII file {file_path}: {e}")

    return slices


def extract_random_slices_from_directory(directory: Path, max_slices: int = 10) -> List[np.ndarray]:
    """
    Извлекает случайную выборку срезов из всех DICOM и NII файлов в директории.

    Args:
        directory: Директория для поиска файлов
        max_slices: Максимальное количество срезов для извлечения

    Returns:
        List[np.ndarray]: Список случайно выбранных срезов
    """
    all_slices = []

    dicom_files = list(directory.rglob("*.dcm")) + list(directory.rglob("*.dicom"))
    nii_files = list(directory.rglob("*.nii"))

    for file_path in directory.rglob("*"):
        if file_path.is_file() and not file_path.suffix:
            if is_probable_dicom_file(file_path):
                dicom_files.append(file_path)

    for file_path in dicom_files:
        slices = extract_slices_from_dicom(file_path)
        all_slices.extend(slices)

    for file_path in nii_files:
        slices = extract_slices_from_nii(file_path)
        all_slices.extend(slices)

    if len(all_slices) <= max_slices:
        return all_slices
    else:
        return random.sample(all_slices, max_slices)


def iterate_dicom_nii_slices(
        base_dirs: Union[Path, List[Path], str, List[str]],
        class_map: Optional[dict] = None,
        recursive: bool = True,
        verbose: bool = False,
        subset: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, int, Path]]:
    """
    Итератор по срезам DICOM и NII файлов с присвоением меток классов.

    Args:
        base_dirs: Базовые директории для поиска
        class_map: Словарь сопоставления папок и меток классов
        recursive: Рекурсивный поиск файлов
        verbose: Вывод прогресса
        subset: Ограничение количества срезов

    Yields:
        Tuple[np.ndarray, int, Path]: Срез, метка класса, путь к файлу
    """
    if class_map is None:
        class_map = {
            'norma_anon': 0,
            'pneumonia_anon': 1,
            'pneumotorax_anon': 2,
            'CT-0': 0,
            'CT-1': 3,
            'CT-2': 3,
            'CT-3': 3,
        }

    if subset is not None and subset <= 0:
        return

    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    base_dirs = [Path(d) for d in base_dirs]
    print(base_dirs)
    all_file_paths = []
    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        for class_folder, label in class_map.items():
            folder_path = base_dir / class_folder
            if not folder_path.is_dir():
                print(f"Warning: {folder_path} does not exist, skipping.")
                continue

            glob_pattern = "**/*.dcm" if recursive else "*.dcm"
            dicom_files = list(folder_path.glob(glob_pattern))
            glob_pattern = "**/*.nii" if recursive else "*.nii"
            nii_files = list(folder_path.glob(glob_pattern))
            glob_pattern = "**/*" if recursive else "*"
            no_ext_files = [p for p in folder_path.glob(glob_pattern) if p.is_file() and p.suffix == '']
            dicom_files.extend(no_ext_files)

            all_file_paths.extend(
                [(p, label, "dcm") for p in dicom_files if p.is_file()]
            )
            all_file_paths.extend(
                [(p, label, "nii") for p in nii_files if p.is_file()]
            )

    count = 0
    iterator = tqdm(all_file_paths, desc="Loading slices", disable=not verbose)

    for file_path, label, ext in iterator:
        if subset is not None and count >= subset:
            break

        try:
            if ext == "dcm":
                img_array = dcm.dcmread(str(file_path)).pixel_array.astype(np.float32)
                if img_array.ndim == 3:
                    for slice_idx in range(img_array.shape[-1]):
                        if subset is not None and count >= subset:
                            return
                        slice_2d = img_array[:, :, slice_idx]
                        if np.max(slice_2d) == 0:
                            continue
                        yield slice_2d, label, file_path
                        count += 1
                else:
                    if np.max(img_array) == 0:
                        continue
                    yield img_array.squeeze(), label, file_path
                    count += 1

            elif ext == "nii":
                img_3d = read_nii(file_path)
                for slice_idx in range(img_3d.shape[-1]):
                    if subset is not None and count >= subset:
                        return
                    slice_2d = img_3d[:, :, slice_idx]
                    if np.max(slice_2d) == 0:
                        continue
                    yield slice_2d, label, file_path
                    count += 1

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue