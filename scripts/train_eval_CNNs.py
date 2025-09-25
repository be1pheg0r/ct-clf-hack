import argparse
from pathlib import Path
import yaml
from datetime import datetime

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ct_clf_hack.models.ModelCNN import ModelCNN


def load_images_from_folders(base_dirs):
    images = []
    labels = []
    class_map = {
        'norma_anon': 0,
        'pneumonia_anon': 1,
        'pneumotorax_anon': 2,
        # 'CT-0': 0,
        # 'CT-1': 3,
        # 'CT-2': 3,
        # 'CT-3': 3,
        # 'CT-4': 4
    }

    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        print(f"Processing directory: {base_dir}")

        for class_folder, label in class_map.items():
            folder_path = base_dir / class_folder
            if not folder_path.is_dir():
                print(f"Warning: {folder_path} does not exist, skipping.")
                continue

            for file_path in folder_path.iterdir():
                if not file_path.is_file():
                    continue

                if file_path.suffix == '.nii':
                    try:
                        nii_img = nib.load(str(file_path))
                        img_data = nii_img.get_fdata()
                        for slice_idx in range(img_data.shape[-1]):
                            slice_2d = img_data[:, :, slice_idx]
                            if np.max(slice_2d) == 0:
                                continue
                            slice_min, slice_max = np.min(slice_2d), np.max(slice_2d)
                            if slice_max > slice_min:
                                slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
                                slice_normalized = (slice_normalized * 255).astype(np.uint8)
                                pil_img = Image.fromarray(slice_normalized).convert("L")
                                images.append(pil_img)
                                labels.append(label)
                    except Exception as e:
                        print(f"Error with file {file_path}: {e}")
                    continue

                try:
                    img_array = dicom.dcmread(str(file_path)).pixel_array
                    if img_array.ndim == 3:
                        for slice_idx in range(img_array.shape[0]):
                            slice_2d = img_array[slice_idx]
                            if np.max(slice_2d) == 0:
                                continue
                            slice_min, slice_max = np.min(slice_2d), np.max(slice_2d)
                            if slice_max > slice_min:
                                slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
                                slice_normalized = (slice_normalized * 255).astype(np.uint8)
                                pil_img = Image.fromarray(slice_normalized).convert("L")
                                images.append(pil_img)
                                labels.append(label)
                    else:
                        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
                        img_array = (img_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array.squeeze()).convert("L")
                        images.append(pil_img)
                        labels.append(label)
                except Exception as e:
                    print(f"Error with file {file_path}: {e}")

    return images, labels

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

    images, labels = load_images_from_folders(data_dirs)
    print(f"Total images loaded: {len(images)}")
    print(f"Class 1: {labels.count(1)}, Class 0: {labels.count(0)}")
    images = list(images)
    labels = list(labels)

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

            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoints_dir / f"{arch}_{dt_str}.pth"
            model.save(str(checkpoint_path))
            print(f"Checkpoint saved to {checkpoint_path}")

            test_dataset = model.transforms
            test_data = [test_dataset(img) for img in test_images]
            test_data = torch.stack(test_data)

            model.model.eval()
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(test_data), args.batch_size):
                    batch = test_data[i:i + args.batch_size].to(model.device).float()
                    outputs = model.model(batch)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)

            accuracy = accuracy_score(test_labels, all_preds)
            precision = precision_score(test_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(test_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, all_preds, average='weighted', zero_division=0)
            results.append({
                'Architecture': arch,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })

            if num_classes == 2:
                cm = confusion_matrix(test_labels, all_preds)
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

            test_dataset = model.transforms
            test_data = [test_dataset(img) for img in test_images]
            test_data = torch.stack(test_data)

            model.model.eval()
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(test_data), args.batch_size):
                    batch = test_data[i:i + args.batch_size].to(model.device).float()
                    outputs = model.model(batch)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)

            accuracy = accuracy_score(test_labels, all_preds)
            precision = precision_score(test_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(test_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, all_preds, average='weighted', zero_division=0)
            results.append({
                'Architecture': arch,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })

            if num_classes == 2:
                cm = confusion_matrix(test_labels, all_preds)
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
