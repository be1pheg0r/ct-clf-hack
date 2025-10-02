import argparse
import logging
import os
from pathlib import Path

from ct_clf_hack.shared.logger_utils import setup_logger
from ct_clf_hack.shared.config_utils import models_config


def download(
        repo_id: str,
        target_path: Path,
) -> bool:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download, login
    except Exception as e:  # ImportError и т.п.
        logging.error("huggingface_hub не установлен: %s", e)
        raise

    # Попробуем snapshot_download (скачивает/копирует папку репо локально)
    try:
        logging.info(f"Используем snapshot_download для {repo_id}")
        local_repo_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
        )
        return True
    except Exception as e:
        logging.exception("Ошибка при hf_hub_download: %s", e)
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install model checkpoints from Hugging Face Hub")
    p.add_argument("--checkpoints-dir", type=str, default="./checkpoints",
                   help="Directory where checkpoints are stored")
    p.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"),
                   help="Hugging Face access token (or set HF_TOKEN env var)")
    p.add_argument("--use-snapshot", action="store_true",
                   help="Use snapshot_download (useful to mirror repo structure)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(__name__)

    download(
        repo_id=models_config()["ResNet50"]["remote"],
        target_path=Path(args.checkpoints_dir)
    )
    if args.verbose:
        logger.info("Checkpoints installed in %s", args.checkpoints_dir)

if __name__ == "__main__":
    main()
