"""
Скрипт для загрузки моделей с Huggingface Hub.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def download_models(models_config: Dict[str, Any], cache_dir: str = "./checkpoints") -> None:
    """
    Загружает модели с Huggingface Hub.

    Args:
        models_config: Конфигурация моделей
        cache_dir: Директория для кэша моделей
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: transformers или huggingface_hub не установлены")
        print("Установите: pip install transformers huggingface_hub")
        sys.exit(1)

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
        "nvidia/mit-b0"
    ]

    successful_downloads = 0
    failed_downloads = 0

    for model_name in huggingface_models:
        try:
            print(f"Загрузка {model_name}...")

            # Загружаем модель и токенайзер
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            except:
                # Не все модели имеют токенайзер
                pass

            print(f"✓ Успешно загружен {model_name}")
            successful_downloads += 1

        except Exception as e:
            print(f"✗ Ошибка при загрузке {model_name}: {e}")
            failed_downloads += 1

    print(f"\nРезультат загрузки:")
    print(f"Успешно: {successful_downloads}")
    print(f"Ошибок: {failed_downloads}")

    if failed_downloads > 0:
        print("\nНекоторые модели не удалось загрузить.")
        print("Проверьте интернет-соединение и доступность моделей.")

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

    # Загружаем конфигурацию моделей
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from ct_clf_hack.shared.config_utils import models_config
        config = models_config()
    except Exception as e:
        print(f"Не удалось загрузить конфигурацию: {e}")
        config = {}

    download_models(config, args.cache_dir)

if __name__ == "__main__":
    main()

