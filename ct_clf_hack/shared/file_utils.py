from pathlib import Path
import numpy as np
import pydicom as dcm
import nibabel as nib
from PIL import Image
from typing import List, Tuple, Optional, Union, Iterator
from tqdm import tqdm


def read_dicom(file_path: Path, apply_rescale: bool = True) -> np.ndarray:
    ds = dcm.dcmread(str(file_path))
    img = ds.pixel_array.astype(np.float32)

    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

    return img


def read_nii(file_path: Path) -> np.ndarray:
    nii_img = nib.load(str(file_path))
    img_data = nii_img.get_fdata().astype(np.float32)
    return img_data


def save_image(img: np.ndarray, file_path: Path):
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
        verbose: bool = False,
        subset: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, int, Path]]:
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