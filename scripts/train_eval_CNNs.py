import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from ct_clf_hack.models.ModelCNN import ModelCNN
from ct_clf_hack.shared.data_utils import (class_map, MainDataset, create_dataloader,
                                           split_dataset)


def evaluate_model(model, test_dataset, num_classes):
    model.model.eval()
    all_preds = []
    all_labels = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(model.device).float()
            outputs = model.model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    result = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }

    if num_classes == 2:
        cm = confusion_matrix(all_labels, all_preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            result.update({
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })

    return result, all_preds


def main():
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

    dataset = MainDataset(dpaths=data_dirs, class_map=class_map, verbose=True)

    labels = [dataset.labels[i].item() for i in range(len(dataset))]
    print(f"Total images loaded: {len(dataset)}")
    print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")

    train_dataset, test_dataset = split_dataset(dataset, val_ratio=0.2)
    print(f"Train/Test split: {len(train_dataset)}/{len(test_dataset)}")

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
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
            print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

            model.load_data(train_loader)
            model.train(epochs=args.epochs)

            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoints_dir / f"{arch}_{dt_str}.pth"
            model.save(str(checkpoint_path))
            print(f"Checkpoint saved to {checkpoint_path}")

            result, all_preds = evaluate_model(model, test_dataset, num_classes)
            result['Architecture'] = arch
            results.append(result)

            print(f"{arch} - Accuracy: {result['Accuracy']:.4f}, Precision: {result['Precision']:.4f}, Recall: {result['Recall']:.4f}, F1: {result['F1_Score']:.4f}")

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

            result, all_preds = evaluate_model(model, test_dataset, num_classes)
            result['Architecture'] = arch
            results.append(result)

            print(f"{arch} - Accuracy: {result['Accuracy']:.4f}, Precision: {result['Precision']:.4f}, Recall: {result['Recall']:.4f}, F1: {result['F1_Score']:.4f}")

    results_df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(results_df.round(4))

if __name__ == "__main__":
    main()
