"""
Модуль пайплайнов для обработки медицинских изображений.
Содержит основную функцию обработки и тестирования модели классификации.
"""

from pathlib import Path
from typing import Any, List, Union, Dict, Optional

import numpy as np
import torch

from ct_clf_hack.models.ModelCNN import ConvSensus, check_checkpoints
from ct_clf_hack.shared.config_utils import models_config
from ct_clf_hack.shared.data_utils import (load_images_from_folders,
                                           prepare_images_for_model,
                                           process_zip_dicom_for_backend)
from ct_clf_hack.shared.logger_utils import setup_logger

logger = setup_logger(__name__)
models_cfg = models_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if not check_checkpoints(models_cfg):
    logger.warning("Some model checkpoints are missing. Ensemble may not work properly.")

ensemble = ConvSensus(
    models_config=models_cfg,
    device=device,
)

def base_process(
        zip_path: Optional[Path] = None,
        dpath: Optional[Path] = None,
        images: Optional[Union[List, Any]] = None,
) -> Dict[str, Any]:
    """
    Основная функция обработки медицинских изображений с помощью модели.

    Args:
        zip_path: Путь к ZIP архиву с DICOM файлами
        dpath: Путь к директории с DICOM файлами
        images: Предварительно подготовленные изображения

    Returns:
        Dict[str, Any]: Результаты анализа с вероятностью патологии и предсказанием
    """
    schema = {
        "path_to_study": zip_path,
        "pathology": None,
        "probability_of_pathology": None,
    }

    try:
        if zip_path is not None and zip_path.exists():
            logger.info(f"Processing zip archive: {zip_path}")
            images, status = process_zip_dicom_for_backend(zip_path, max_slices=10, delete=False)
            if not images:
                logger.warning(f"No images extracted from zip: {status}")
                return schema
            images = prepare_images_for_model(images)

        elif dpath is not None and images is None:
            logger.info(f"Processing directory: {dpath}")
            images, _ = load_images_from_folders([dpath], class_map={dpath.name: 0})

        elif images is not None and dpath is None and zip_path is None:
            logger.info("Processing provided images")
            images = prepare_images_for_model(images)

        else:
            logger.error("Either zip_path, dpath or images must be provided (but only one).")
            return schema

        logger.info(f"Total images loaded: {len(images)}")

        if len(images) == 0:
            logger.info("No images found for processing.")
            return schema
        probabilities, predictions = ensemble.predict(images, batch_size=len(images))
        if len(probabilities) > 0:
            probabilities = np.mean(probabilities, axis=0)
            max_prob = float(np.max(probabilities))
            prediction = np.argmax(probabilities)
        else:
            max_prob = 0.0
            prediction = 0

        max_prob = max_prob if prediction != 0 else 1.0 - max_prob

        schema["pathology"] = prediction
        schema["probability_of_pathology"] = max_prob

        logger.info(f"Prediction: {schema['pathology']}, Probability: {schema['probability_of_pathology']:.4f}")
        return schema

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return schema