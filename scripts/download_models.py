"""
Скрипт для загрузки моделей с Huggingface Hub.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def download_models(cache_dir: str = "./checkpoints") -> None:
    """
    Загружает модели с Huggingface Hub.

    Args:
        cache_dir: Директория для кэша моделей
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: transformers или huggingface_hub не установлены")
        print("Установите: pip install transformers huggingface_hub")
        return

    # Создаем директорию для моделей
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Устанавливаем переменную окружения для Huggingface
    os.environ['HF_HOME'] = str(cache_path)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_path)

    print(f"Загрузка моделей в: {cache_path}")

    # Модели для загрузки (можно настроить)
    huggingface_models = [
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
        "facebook/convnext-base-224-22k",
        "microsoft/swin-base-patch4-window7-224",
    ]

    successful_downloads = 0
    failed_downloads = 0

    for model_name in huggingface_models:
        try:
            print(f"Загрузка {model_name}...")

            # Загружаем модель
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            print(f"✓ Успешно загружен {model_name}")
            successful_downloads += 1

        except Exception as e:
            print(f"✗ Ошибка при загрузке {model_name}: {e}")
            failed_downloads += 1

    print(f"\nРезультат загрузки:")
    print(f"Успешно: {successful_downloads}")
    print(f"Ошибок: {failed_downloads}")

def main():
    """Основная функция."""
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка моделей с Huggingface")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./checkpoints",
        help="Директория для кэша моделей"
    )

    args = parser.parse_args()
    download_models(args.cache_dir)

if __name__ == "__main__":
    main()
