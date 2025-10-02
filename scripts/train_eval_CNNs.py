"""
Скрипт для обучения и оценки различных архитектур CNN на изображениях КТ.
"""

import argparse
from pathlib import Path
import yaml
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ct_clf_hack.models.ModelCNN import ModelCNN
from ct_clf_hack.shared.data_utils import load_images_from_folders
from ct_clf_hack.shared.config_utils import class_map_config


def main():
    """
    Скрипт для обучения и оценки различных архитектур CNN на изображениях КТ.
    ПРИМ.: class_map.yaml должен быть настроен заранее. (configs/class_map.yaml)
    """
    parser = argparse.ArgumentParser(description="Train and evaluate CNN models on CT scan images.")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True, help="Path(s) to the dataset directory containing class subdirectories.")
    parser.add_argument("--rs", type=int, default=52, help="Random seed")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/", help="Directory to save/load model checkpoints.")
    parser.add_argument("--train", type=int, choices=[0, 1], required=True, help="Train models (1) or only evaluate (0).")
    parser.add_argument("--config_yaml", type=str, default="configs/models.yaml", help="YAML file with model paths for evaluation.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and inference.")
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs]
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    class_map = class_map_config()

    images, labels = load_images_from_folders(data_dirs, class_map)
    print(f"Total images loaded: {len(images)}")
    print(f"Class 1: {labels.count(1)}, Class 0: {labels.count(0)}")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=args.rs, stratify=labels, shuffle=True
    )
    num_classes = len(set(labels))

    architectures = ["Inception_V3", "ResNet50", "DenseNet121", "ConvNeXt_Tiny"]
    results = []

    if args.train == 1:
        for arch in architectures:
            print(f"\n=== Training {arch} ===")
            model = ModelCNN(arch, num_classes=num_classes)
            print(f"Using model architecture: {model.model_name}")
            print(f"Number of classes: {model.num_classes}")
            print(f"Using device: {model.device}")
            print(f"Train: {len(train_images)}, Test: {len(test_images)}")

            model.load_data(train_images, train_labels)
            model.dataloader = torch.utils.data.DataLoader(
                model.dataloader.dataset, batch_size=args.batch_size, shuffle=True
            )
            model.train(epochs=args.epochs)

            save_path = checkpoints_dir / f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            model.save(str(save_path))

            test_preds = model.predict(test_images, batch_size=args.batch_size)

            accuracy = accuracy_score(test_labels, test_preds)
            precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
            recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
            results.append({
                'Architecture': arch,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })

            if num_classes == 2:
                cm = confusion_matrix(test_labels, test_preds)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    results[-1].update({
                        'TP': tp,
                        'FP': fp,
                        'FN': fn,
                        'TN': tn
                    })

            print(f"{arch} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    else:
        config_yaml = Path(args.config_yaml)
        if not config_yaml.exists():
            print(f"Config yaml not found: {config_yaml}")
            return

        with open(config_yaml, "r") as f:
            arch_paths = yaml.safe_load(f)

        for arch in architectures:
            if arch not in arch_paths:
                print(f"Model path for {arch} not found in config yaml, skipping.")
                continue
            model_path = arch_paths[arch]
            print(f"\n=== Evaluating {arch} ===")
            model = ModelCNN(arch, num_classes=num_classes)
            model.load(model_path)

            test_preds = model.predict(test_images, batch_size=args.batch_size)

            accuracy = accuracy_score(test_labels, test_preds)
            precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
            recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
            results.append({
                'Architecture': arch,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })

            if num_classes == 2:
                cm = confusion_matrix(test_labels, test_preds)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    results[-1].update({
                        'TP': tp,
                        'FP': fp,
                        'FN': fn,
                        'TN': tn
                    })

            print(f"{arch} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    results_df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
