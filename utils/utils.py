from pathlib import Path
import numpy as np
import pydicom as dcm
import nibabel as nib
from PIL import Image
from typing import List, Tuple, Optional, Union, Iterator
from tqdm import tqdm


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


def read_nii(file_path: Path) -> np.ndarray:
    """
    Читает .nii файл и возвращает 3D numpy-массив.

    Args:
        file_path (Path): путь к файлу .nii

    Returns:
        np.ndarray: 3D изображение (H, W, D), float32
    """
    nii_img = nib.load(str(file_path))
    img_data = nii_img.get_fdata().astype(np.float32)
    return img_data


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

    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img)

    pil_img.save(str(file_path))


def iterate_dicom_nii_slices(
    base_dirs: Union[Path, List[Path], str, List[str]],
    class_map: Optional[dict] = None,
    recursive: bool = True,
    show_progress: bool = False
) -> Iterator[Tuple[np.ndarray, int, Path]]:
    """
    Итеративно проходит по директориям, читает DICOM и NIfTI файлы,
    извлекает 2D срезы и возвращает их вместе с меткой и путем к файлу.

    Args:
        base_dirs: одна или несколько директорий для поиска файлов.
        class_map: словарь соответствия имени папки классу, например,
                   {'norma_anon': 0, 'pneumonia_anon': 1}.
                   Если None, используется стандартный словарь.
        recursive: искать ли файлы рекурсивно в поддиректориях.
        show_progress: показывать ли прогресс-бар (медленно, если True).

    """
    if class_map is None:
        class_map = {
            'norma_anon': 0,
            'pneumonia_anon': 1,
            'pneumotorax_anon': 2,
        }

    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    base_dirs = [Path(d) for d in base_dirs]

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


            all_file_paths.extend(
                [(p, label, "dcm") for p in dicom_files if p.is_file()]
            )
            all_file_paths.extend(
                [(p, label, "nii") for p in nii_files if p.is_file()]
            )


    iterator = tqdm(all_file_paths, desc="Loading slices", disable=not show_progress)
    for file_path, label, ext in iterator:
        try:
            if ext == "dcm":
                img_array = dcm.dcmread(str(file_path)).pixel_array.astype(np.float32)
                # применяем RescaleSlope/Intercept, если нужно
                # slope = float(getattr(ds, "RescaleSlope", 1.0))
                # intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                # img_array = img_array * slope + intercept

                if img_array.ndim == 3:
                    # Предполагаем, что это 3D (H, W, Slices)
                    for slice_idx in range(img_array.shape[-1]):
                        slice_2d = img_array[:, :, slice_idx]
                        if np.max(slice_2d) == 0:
                            continue
                        yield slice_2d, label, file_path
                else:
                    # 2D изображение
                    if np.max(img_array) == 0:
                        continue
                    yield img_array.squeeze(), label, file_path

            elif ext == "nii":
                img_3d = read_nii(file_path)
                # img_3d = img_3d.astype(np.float32) # уже в read_nii
                for slice_idx in range(img_3d.shape[-1]):
                    slice_2d = img_3d[:, :, slice_idx]
                    if np.max(slice_2d) == 0:
                        continue
                    yield slice_2d, label, file_path

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue # Пропускаем файл с ошибкой