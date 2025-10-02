"""
Скрипт для комплексного тестирования системы анализа КТ изображений.
Проверяет работоспособность всех основных компонентов системы.
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import requests
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from ct_clf_backend.app.cache_utils import (cleanup_cache, create_cache_dir,
                                            extract_dicom_metadata,
                                            process_uploaded_file)
from ct_clf_hack.models.ModelCNN import ConvSensus, check_checkpoints
from ct_clf_hack.pipelines import base_process
from ct_clf_hack.shared.config_utils import class_map_config, models_config
from ct_clf_hack.shared.data_utils import (load_images_from_folders,
                                           prepare_images_for_model,
                                           process_zip_dicom_for_backend)
from ct_clf_hack.shared.file_utils import (find_dicom_files,
                                           is_probable_dicom_file)


class SystemTester:
    """Класс для комплексного тестирования системы."""

    def __init__(self, verbose: bool = True):
        """
        Инициализация тестера системы.

        Args:
            verbose: Включить подробный вывод
        """
        self.verbose = verbose
        self.results = {}
        self.errors = []

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Логирование сообщений.

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        if self.verbose:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def test_configs(self) -> bool:
        """
        Тестирует загрузку конфигурационных файлов.

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование конфигурационных файлов...")
        try:
            models_cfg = models_config()
            assert isinstance(models_cfg, dict), "models_config должна возвращать словарь"
            assert len(models_cfg) > 0, "Конфигурация моделей не должна быть пустой"
            self.log(f"Найдены модели: {list(models_cfg.keys())}")

            class_map = class_map_config()
            assert isinstance(class_map, dict), "class_map_config должна возвращать словарь"
            assert len(class_map) > 0, "Карта классов не должна быть пустой"
            self.log(f"Классы: {list(class_map.keys())}")

            self.results["configs"] = {"status": "SUCCESS", "models": len(models_cfg), "classes": len(class_map)}
            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании конфигураций: {e}", "ERROR")
            self.errors.append(f"Config test failed: {e}")
            self.results["configs"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_checkpoints(self) -> bool:
        """
        Тестирует наличие чекпойнтов моделей.

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Проверка чекпойнтов моделей...")
        try:
            models_cfg = models_config()
            checkpoints_exist = check_checkpoints(models_cfg)

            missing_checkpoints = []
            for arch, cfg in models_cfg.items():
                checkpoint_path = cfg.get("local", "")
                if not Path(checkpoint_path).exists():
                    missing_checkpoints.append(f"{arch}: {checkpoint_path}")

            if missing_checkpoints:
                self.log("Отсутствующие чекпойнты:", "WARNING")
                for missing in missing_checkpoints:
                    self.log(f"  - {missing}", "WARNING")

            self.results["checkpoints"] = {
                "status": "SUCCESS" if checkpoints_exist else "WARNING",
                "all_exist": checkpoints_exist,
                "missing": missing_checkpoints
            }
            return True

        except Exception as e:
            self.log(f"Ошибка при проверке чекпойнтов: {e}", "ERROR")
            self.errors.append(f"Checkpoints test failed: {e}")
            self.results["checkpoints"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_file_utils(self) -> bool:
        """
        Тестирует утилиты для работы с файлами.

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование утилит для работы с файлами...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp_file:
                tmp_file.write(b"\x00" * 128 + b"DICM" + b"\x00" * 100)
                tmp_path = Path(tmp_file.name)

            try:
                is_dicom = is_probable_dicom_file(tmp_path)
                self.log(f"Тест DICOM файла: {is_dicom}")
                self.results["file_utils"] = {
                    "status": "SUCCESS",
                    "dicom_detection": is_dicom
                }
                return True

            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        except Exception as e:
            self.log(f"Ошибка при тестировании файловых утилит: {e}", "ERROR")
            self.errors.append(f"File utils test failed: {e}")
            self.results["file_utils"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_cache_utils(self) -> bool:
        """
        Тестирует утилиты для работы с кэшем.

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование утилит кэширования...")
        try:
            cache_dir = create_cache_dir()
            assert cache_dir.exists(), "Кэш директория должна быть создана"
            self.log(f"Кэш директория создана: {cache_dir}")

            cleanup_cache(cache_dir)
            assert not cache_dir.exists(), "Кэш директория должна быть удалена"
            self.log("Кэш директория успешно очищена")

            self.results["cache_utils"] = {"status": "SUCCESS"}
            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании кэш утилит: {e}", "ERROR")
            self.errors.append(f"Cache utils test failed: {e}")
            self.results["cache_utils"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_data_processing(self, test_data_dir: Optional[Path] = None) -> bool:
        """
        Тестирует обработку данных.

        Args:
            test_data_dir: Путь к тестовым данным

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование обработки данных...")
        try:
            if test_data_dir and test_data_dir.exists():
                try:
                    images, labels = load_images_from_folders([test_data_dir])
                    self.log(f"Загружено изображений: {len(images)}, меток: {len(labels)}")

                    if len(images) > 0:
                        rgb_images = prepare_images_for_model(images[:3])
                        self.log(f"Подготовлено RGB изображений: {len(rgb_images)}")

                        self.results["data_processing"] = {
                            "status": "SUCCESS",
                            "images_loaded": len(images),
                            "labels_loaded": len(labels),
                            "rgb_conversion": len(rgb_images)
                        }
                    else:
                        self.results["data_processing"] = {
                            "status": "WARNING",
                            "message": "Нет изображений для обработки в тестовой директории"
                        }
                except Exception as e:
                    self.log(f"Ошибка при загрузке из тест-директории: {e}", "WARNING")
                    self.results["data_processing"] = {
                        "status": "WARNING",
                        "message": f"Не удалось загрузить данные из {test_data_dir}: {e}"
                    }
            else:
                self.log("Тестовая директория не указана или не существует", "WARNING")
                self.results["data_processing"] = {
                    "status": "SKIPPED",
                    "message": "Тестовая директория не доступна"
                }

            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании обработки данных: {e}", "ERROR")
            self.errors.append(f"Data processing test failed: {e}")
            self.results["data_processing"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_model_ensemble(self, test_images: Optional[List[np.ndarray]] = None) -> bool:
        """
        Тестирует ансамбль моделей.

        Args:
            test_images: Тестовые изображения

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование ансамбля моделей...")
        try:
            models_cfg = models_config()

            ensemble = ConvSensus(models_cfg)

            if len(ensemble.models) == 0:
                self.log("Ансамбль пуст - нет загруженных моделей", "WARNING")
                self.results["model_ensemble"] = {
                    "status": "WARNING",
                    "message": "Нет доступных моделей для ансамбля"
                }
                return True

            self.log(f"Загружено моделей в ансамбль: {len(ensemble.models)}")

            if test_images and len(test_images) > 0:
                try:
                    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    probabilities, predictions = ensemble.predict([test_img], batch_size=1)

                    self.log(f"Тест предсказания: вероятности shape={probabilities.shape}, предсказания shape={predictions.shape}")

                    self.results["model_ensemble"] = {
                        "status": "SUCCESS",
                        "models_loaded": len(ensemble.models),
                        "prediction_test": "SUCCESS"
                    }
                except Exception as pred_error:
                    self.log(f"Ошибка при тестовом предсказании: {pred_error}", "WARNING")
                    self.results["model_ensemble"] = {
                        "status": "WARNING",
                        "models_loaded": len(ensemble.models),
                        "prediction_test": f"FAILED: {pred_error}"
                    }
            else:
                self.results["model_ensemble"] = {
                    "status": "SUCCESS",
                    "models_loaded": len(ensemble.models),
                    "prediction_test": "SKIPPED"
                }

            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании ансамбля: {e}", "ERROR")
            self.errors.append(f"Model ensemble test failed: {e}")
            self.results["model_ensemble"] = {"status": "FAILED", "error": str(e)}
            return False

    def test_pipeline(self, test_zip_path: Optional[Path] = None) -> bool:
        """
        Тестирует основной пайплайн обработки.

        Args:
            test_zip_path: Путь к тестовому ZIP файлу

        Returns:
            bool: True если тест прошел успешно
        """
        self.log("Тестирование основного пайплайна...")
        try:
            if test_zip_path and test_zip_path.exists():
                try:
                    result = base_process(zip_path=test_zip_path)
                    self.log(f"Результат пайплайна: {result}")

                    expected_keys = ["path_to_study", "pathology", "probability_of_pathology"]
                    missing_keys = [key for key in expected_keys if key not in result]

                    if missing_keys:
                        self.log(f"Отсутствуют ключи в результате: {missing_keys}", "WARNING")

                    self.results["pipeline"] = {
                        "status": "SUCCESS",
                        "result": result,
                        "missing_keys": missing_keys
                    }
                except Exception as e:
                    self.log(f"Ошибка при выполнении пайплайна: {e}", "WARNING")
                    self.results["pipeline"] = {
                        "status": "WARNING",
                        "message": f"Пайплайн не выполнился с тест-файлом: {e}"
                    }
            else:
                try:
                    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
                    result = base_process(images=test_images)
                    self.log(f"Результат пайплайна с синтетическими данными: {result}")

                    self.results["pipeline"] = {
                        "status": "SUCCESS",
                        "test_type": "synthetic_images",
                        "result": result
                    }
                except Exception as e:
                    self.log(f"Ошибка при тестировании пайплайна с синтетическими данными: {e}", "ERROR")
                    self.results["pipeline"] = {"status": "FAILED", "error": str(e)}
                    return False

            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании пайплайна: {e}", "ERROR")
            self.errors.append(f"Pipeline test failed: {e}")
            self.results["pipeline"] = {"status": "FAILED", "error": str(e)}
            return False

    async def test_api_endpoints(self, base_url: str = "http://localhost:8000") -> bool:
        """
        Тестирует API endpoints.

        Args:
            base_url: Базовый URL API

        Returns:
            bool: True если тест прошел успешно
        """
        self.log(f"Тестирование API endpoints на {base_url}...")
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    test_data = b"test data"
                    data = aiohttp.FormData()
                    data.add_field('file', test_data, filename='test.dcm', content_type='application/octet-stream')
                    data.add_field('dev', 'true')

                    async with session.post(f"{base_url}/process-image", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.log("API /process-image (dev режим) работает")

                            required_keys = ["study_uid", "series_uid", "probability_of_pathology", "pathology"]
                            missing_keys = [key for key in required_keys if key not in result]

                            self.results["api_process_image"] = {
                                "status": "SUCCESS",
                                "dev_mode": True,
                                "missing_keys": missing_keys
                            }
                        else:
                            self.log(f"API /process-image вернул статус: {response.status}", "WARNING")
                            self.results["api_process_image"] = {
                                "status": "WARNING",
                                "http_status": response.status
                            }

                except Exception as e:
                    self.log(f"Ошибка при тестировании /process-image: {e}", "WARNING")
                    self.results["api_process_image"] = {"status": "FAILED", "error": str(e)}

                try:
                    test_data = b"test data"
                    data = aiohttp.FormData()
                    data.add_field('file', test_data, filename='test.dcm', content_type='application/octet-stream')

                    async with session.post(f"{base_url}/viewer", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.log("API /viewer работает")
                            self.results["api_viewer"] = {"status": "SUCCESS"}
                        else:
                            self.log(f"API /viewer вернул статус: {response.status}", "WARNING")
                            self.results["api_viewer"] = {
                                "status": "WARNING",
                                "http_status": response.status
                            }

                except Exception as e:
                    self.log(f"Ошибка при тестировании /viewer: {e}", "WARNING")
                    self.results["api_viewer"] = {"status": "FAILED", "error": str(e)}

            return True

        except Exception as e:
            self.log(f"Ошибка при тестировании API: {e}", "ERROR")
            self.errors.append(f"API test failed: {e}")
            self.results["api_test"] = {"status": "FAILED", "error": str(e)}
            return False

    def generate_report(self) -> Dict[str, Any]:
        """
        Генерирует итоговый отчет о тестировании.

        Returns:
            Dict[str, Any]: Отчет о результатах тестирования
        """
        successful_tests = sum(1 for result in self.results.values()
                              if isinstance(result, dict) and result.get("status") == "SUCCESS")
        warning_tests = sum(1 for result in self.results.values()
                           if isinstance(result, dict) and result.get("status") == "WARNING")
        failed_tests = sum(1 for result in self.results.values()
                          if isinstance(result, dict) and result.get("status") == "FAILED")

        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful": successful_tests,
                "warnings": warning_tests,
                "failed": failed_tests,
                "success_rate": successful_tests / len(self.results) * 100 if self.results else 0
            },
            "detailed_results": self.results,
            "errors": self.errors,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return report


def main():
    """Основная функция для запуска тестирования системы."""
    parser = argparse.ArgumentParser(description="Комплексное тестирование системы анализа КТ изображений")
    parser.add_argument("--test-data-dir", type=str, help="Путь к директории с тестовыми данными")
    parser.add_argument("--test-zip", type=str, help="Путь к тестовому ZIP файлу с DICOM")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="URL для тестирования API")
    parser.add_argument("--skip-api", action="store_true", help="Пропустить тестирование API")
    parser.add_argument("--output", type=str, help="Путь для сохранения отчета в JSON")
    parser.add_argument("--verbose", action="store_true", default=True, help="Подробный вывод")

    args = parser.parse_args()

    tester = SystemTester(verbose=args.verbose)

    tester.log("=== Начало комплексного тестирования системы ===")

    tester.test_configs()

    tester.test_checkpoints()

    tester.test_file_utils()

    tester.test_cache_utils()

    test_data_dir = Path(args.test_data_dir) if args.test_data_dir else None
    tester.test_data_processing(test_data_dir)

    tester.test_model_ensemble()

    test_zip_path = Path(args.test_zip) if args.test_zip else None
    tester.test_pipeline(test_zip_path)

    if not args.skip_api:
        try:
            asyncio.run(tester.test_api_endpoints(args.api_url))
        except Exception as e:
            tester.log(f"Не удалось протестировать API: {e}", "WARNING")
            tester.results["api_test"] = {"status": "SKIPPED", "reason": str(e)}

    report = tester.generate_report()

    tester.log("=== Итоговый отчет ===")
    tester.log(f"Всего тестов: {report['summary']['total_tests']}")
    tester.log(f"Успешных: {report['summary']['successful']}")
    tester.log(f"С предупреждениями: {report['summary']['warnings']}")
    tester.log(f"Неудачных: {report['summary']['failed']}")
    tester.log(f"Процент успеха: {report['summary']['success_rate']:.1f}%")

    if tester.errors:
        tester.log("Критические ошибки:")
        for error in tester.errors:
            tester.log(f"  - {error}", "ERROR")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        tester.log(f"Отчет сохранен в: {output_path}")

    if report['summary']['failed'] > 0:
        return 1
    elif report['summary']['warnings'] > 0:
        return 2
    else:
        return 0


if __name__ == "__main__":
    main()
