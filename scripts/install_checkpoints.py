import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Настройки по умолчанию
MIN_REASONABLE_SIZE = 1024  # bytes - минимальный размер файла, считаем валидным


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_file_with_hf(
    repo_id: str,
    filename: str,
    target_path: Path,
    token: Optional[str] = None,
    use_snapshot: bool = False,
    force_download: bool = False,
) -> bool:
    """Скачивает один файл из репозитория Hugging Face.

    Возвращает True если файл успешно доставлен в target_path и имеет адекватный размер.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download, login
    except Exception as e:  # ImportError и т.п.
        logging.error("huggingface_hub не установлен: %s", e)
        raise

    # Авторизация (безопасно — login хранит токен в окружении)
    if token:
        try:
            # login просто установит токен локально для клиента
            login(token)
            logging.debug("Аутентификация через переданный токен выполнена")
        except Exception:
            # не критично падать на login; hf_hub_download примет token=token
            logging.debug("Не удалось выполнить login(token) — продолжим, передав token в вызовы")

    # Попробуем snapshot_download (скачивает/копирует папку репо локально)
    if use_snapshot:
        try:
            logging.info("Используем snapshot_download для %s (фильтр: %s)", repo_id, filename)
            local_repo_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=[filename],
                token=token,
                local_dir=None,
                ignore_patterns=None,
            )
            candidate = Path(local_repo_dir) / filename
            if candidate.exists():
                # Копируем файл в целевую папку
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target_path)
                logging.debug("Скопирован %s -> %s", candidate, target_path)
            else:
                raise FileNotFoundError(f"Файл {filename} не найден в snapshot {local_repo_dir}")
        except Exception as e:
            logging.exception("snapshot_download не удался: %s", e)
            return False

    else:
        # Используем hf_hub_download для отдельного файла
        try:
            logging.info("hf_hub_download: repo=%s file=%s", repo_id, filename)
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_dir=None,
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
            cached_path = Path(cached_path)
            if not cached_path.exists():
                raise FileNotFoundError(f"hf_hub_download вернул несуществующий путь: {cached_path}")

            # Копируем/перемещаем файл в целевую директорию
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, target_path)
            logging.debug("Скопирован %s -> %s", cached_path, target_path)

        except Exception as e:
            logging.exception("Ошибка при hf_hub_download: %s", e)
            return False

    # Небольшая проверка размера
    try:
        size = target_path.stat().st_size
        if size < MIN_REASONABLE_SIZE:
            logging.warning("Скачанный файл слишком мал (%d bytes): %s", size, target_path)
            return False
        logging.info("Файл загружен и проверен: %s (%d bytes)", target_path, size)
        return True
    except Exception as e:
        logging.exception("Не удалось проверить файл %s: %s", target_path, e)
        return False


def install_model_checkpoints(
    checkpoints_dir: str = "./checkpoints",
    force: bool = False,
    token: Optional[str] = None,
    use_snapshot: bool = False,
) -> Dict[str, Any]:
    """Устанавливает чекпойнты по конфигу models_config().

    Возвращает словарь-отчет с успехами/ошибками.
    """
    from ct_clf_hack.shared.config_utils import models_config

    results = {
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    cfg = models_config()
    logging.info("Загружено %d записей конфигурации модели", len(cfg))

    base_dir = Path(checkpoints_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_cfg in cfg.items():
        local = model_cfg.get("local", "")
        remote = model_cfg.get("remote", "")

        if not local or not remote:
            logging.warning("Пропускаем %s — неполная конфигурация", model_name)
            results["skipped"].append((model_name, "incomplete config"))
            continue

        # Целевой путь
        target_path = Path(local)
        if not target_path.is_absolute():
            target_path = base_dir.joinpath(Path(local).name) if Path(local).parent == Path(".") else Path(local)

        # Если файл есть и не форсим — пропускаем
        if target_path.exists() and not force:
            size = target_path.stat().st_size
            logging.info("Пропускаем %s — уже есть %s (%d bytes)", model_name, target_path, size)
            results["skipped"].append((model_name, "exists", size))
            continue

        # Если форс — удаляем старый файл
        if target_path.exists() and force:
            try:
                target_path.unlink()
                logging.info("Удален старый файл для %s: %s", model_name, target_path)
            except Exception:
                logging.exception("Не удалось удалить старый файл %s", target_path)

        # Имя файла (в репо ожидается именно такое имя)
        filename = Path(local).name

        ok = download_file_with_hf(
            repo_id=remote,
            filename=filename,
            target_path=target_path,
            token=token,
            use_snapshot=use_snapshot,
            force_download=force,
        )

        if ok:
            results["successful"].append((model_name, str(target_path)))
        else:
            # как fallback — создадим пустой файл-заглушку, если это разрешено
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.touch()
                logging.warning("Создан файл-заглушка: %s", target_path)
            except Exception:
                logging.exception("Не удалось создать заглушку %s", target_path)
            results["failed"].append((model_name, str(target_path)))

    # Итогный лог
    logging.info("Успешно: %d, Пропущено: %d, Ошибок: %d",
                 len(results["successful"]), len(results["skipped"]), len(results["failed"]))

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install model checkpoints from Hugging Face Hub")
    p.add_argument("--checkpoints-dir", type=str, default="./checkpoints", help="Directory where checkpoints are stored")
    p.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env var)")
    p.add_argument("--use-snapshot", action="store_true", help="Use snapshot_download (useful to mirror repo structure)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        report = install_model_checkpoints(
            checkpoints_dir=args.checkpoints_dir,
            force=args.force,
            token=args.token,
            use_snapshot=args.use_snapshot,
        )
        logging.info("Отчет: %s", report)
    except Exception:
        logging.exception("Неожиданная ошибка при установке чекпойнтов")
        sys.exit(2)


if __name__ == "__main__":
    main()
