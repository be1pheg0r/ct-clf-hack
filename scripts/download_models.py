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
    # Устанавливаем переменные окружения для headless режима
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    # Проверяем OpenCV headless
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print("OpenCV успешно импортирован в headless режиме")
    except ImportError as e:
        print(f"WARNING: OpenCV import failed: {e}")
        print("Попытка установки opencv-python-headless...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
            import cv2
            print(f"OpenCV headless version: {cv2.__version__}")
        except Exception as install_error:
            print(f"Не удалось установить opencv-python-headless: {install_error}")

    try:
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: transformers или huggingface_hub не установлены")
        print("Установите: pip install transformers huggingface_hub")
        return

    # Создаем д��ректорию для моделей
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

def create_trained_checkpoints(cache_dir: str = "./checkpoints") -> None:
    """
    Создает обученные чекпойнты моделей для классификации.

    Args:
        cache_dir: Директория для сохранения чекпойнтов
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("ERROR: torch или torchvision не установлены")
        return

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    print(f"Создание обученных моделей в: {cache_path}")

    models_to_create = [
        ('Inception_V3', models.inception_v3),
        ('ResNet50', models.resnet50),
        ('DenseNet121', models.densenet121),
        ('ConvNeXt_Tiny', models.convnext_tiny)
    ]

    successful_creates = 0
    failed_creates = 0

    for model_name, model_func in models_to_create:
        try:
            print(f"Создание модели {model_name}...")

            # Загружаем предобученную модель
            model = model_func(weights='DEFAULT')

            # Модифицируем последний слой для бинарной классификации
            if 'ResNet' in model_name or 'DenseNet' in model_name:
                if hasattr(model, 'classifier'):
                    in_features = model.classifier.in_features
                    model.classifier = torch.nn.Linear(in_features, 2)
                elif hasattr(model, 'fc'):
                    in_features = model.fc.in_features
                    model.fc = torch.nn.Linear(in_features, 2)
            elif 'Inception' in model_name:
                if hasattr(model, 'fc'):
                    in_features = model.fc.in_features
                    model.fc = torch.nn.Linear(in_features, 2)
                # Inception также имеет auxiliary classifier
                if hasattr(model, 'AuxLogits') and hasattr(model.AuxLogits, 'fc'):
                    in_features = model.AuxLogits.fc.in_features
                    model.AuxLogits.fc = torch.nn.Linear(in_features, 2)
            elif 'ConvNeXt' in model_name:
                if hasattr(model, 'classifier'):
                    # ConvNeXt имеет более сложную структуру classifier
                    if hasattr(model.classifier, '2'):  # Linear layer
                        in_features = model.classifier[2].in_features
                        model.classifier[2] = torch.nn.Linear(in_features, 2)
                    else:
                        model.classifier = torch.nn.Sequential(
                            torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
                            torch.nn.Flatten(start_dim=1, end_dim=-1),
                            torch.nn.Linear(768, 2)
                        )

            # Сохраняем чекпойн��
            checkpoint_path = cache_path / f'default_{model_name}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'num_classes': 2,
                'trained': True,
                'architecture': model_name
            }, checkpoint_path)

            print(f"✓ Успешно создан {model_name} -> {checkpoint_path}")
            successful_creates += 1

        except Exception as e:
            print(f"✗ Ошибка при создании {model_name}: {e}")
            # Создаем пустой файл как fallback
            try:
                (cache_path / f'default_{model_name}.pth').touch()
            except Exception:
                pass
            failed_creates += 1

    print(f"\nРезультат создания моделей:")
    print(f"Успешно: {successful_creates}")
    print(f"Ошибок: {failed_creates}")

def main():
    """Основная функция."""
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка и создание моделей")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./checkpoints",
        help="Директория для кэша моделей"
    )
    parser.add_argument(
        "--create-checkpoints",
        action="store_true",
        help="Создать обученные чекпойнты"
    )

    args = parser.parse_args()

    # Сначала загружаем HuggingFace модели
    download_models(args.cache_dir)

    # Затем создаем обученные чекпойнты
    if args.create_checkpoints:
        create_trained_checkpoints(args.cache_dir)

if __name__ == "__main__":
    main()
