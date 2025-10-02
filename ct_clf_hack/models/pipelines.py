import torch

from ct_clf_hack.models.ModelCNN import ConvSensus, CNNLoader
from ct_clf_hack.shared.data_utils import MainDataset
from ct_clf_hack.shared.logger_utils import default_logger
from ct_clf_hack.shared.config_utils import class_map, models_config
from ct_clf_hack.shared.file_utils import read_dicom, read_nii
from pathlib import Path

import numpy as np
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_cfg = models_config()
    model_names = list(models_cfg.keys())
    default_logger.info(f"Available models: {model_names}")

    sample_dir = Path("data/uncompressed/base/norma_anon")
    file_paths = list(sample_dir.glob("**/*"))[:20]
    data = []
    loop = tqdm(file_paths, desc="Loading files")
    if ".nii" in [p.suffix for p in file_paths]:
        for p in loop:
            if p.suffix == ".nii":
                data.append([read_nii(p)])
    else:
        for p in loop:
            if p.suffix in [".dcm", ".dicom", ""]:
                try:
                    img = read_dicom(p)
                    if img is not None:
                        data.append([img])
                except Exception as e:
                    default_logger.warning(f"Failed to read {p}: {e}")

    data = np.array(data)

    models_loader = CNNLoader(models_config=models_cfg, device=device)

    dataset = MainDataset(data=data, verbose=True)
    default_logger.info(f"Total images loaded: {len(dataset)}")
    default_logger.info(f"Data shape after loading: {data.shape}")

    convsensus = ConvSensus(
        models=models_loader.load_models(),
        num_classes=models_loader.num_classes
    )


    import pandas as pd
    probs = convsensus.predict(dataset, labels=False)
    print(probs)


if __name__ == "__main__":
    main()

