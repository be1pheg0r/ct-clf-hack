import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import silhouette_score

from ct_clf_hack.models.KNN import KNNSeparator
from ct_clf_hack.shared.data_utils import class_map, MainDataset, split_dataset


def evaluate_clustering(knn_separator, X_data, k_values):
    best_k, elbow_results = knn_separator.elbow_method(X_data, max_k=max(k_values))

    silhouette_scores = []
    for k in k_values:
        if k > 1:
            knn_temp = KNNSeparator(k=k)
            cluster_labels = knn_temp.fit_predict(X_data)
            sil_score = silhouette_score(X_data, cluster_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)

    best_sil_k = k_values[np.argmax(silhouette_scores)]

    return {
        'best_k_elbow': best_k,
        'best_k_silhouette': best_sil_k,
        'elbow_results': elbow_results,
        'silhouette_scores': dict(zip(k_values, silhouette_scores))
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate KNN clustering on CT scan images.")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True,
                        help="Path(s) to the dataset directory containing class subdirectories.")
    parser.add_argument("--rs", type=int, default=52, help="Random seed")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/",
                        help="Directory to save/load model checkpoints.")
    parser.add_argument("--train", type=int, choices=[0, 1], required=True,
                        help="Train models (1) or only evaluate (0).")
    parser.add_argument("--config_yaml", type=str, default="configs/knn_models.yaml",
                        help="YAML file with model paths for evaluation.")
    parser.add_argument("--k_clusters", type=int, nargs='+', default=[2, 4, 6, 8, 10],
                        help="K values for KNN clustering.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use a subset of the data for quick testing (number of samples).")
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs]
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset = MainDataset(dpaths=data_dirs, class_map=class_map, verbose=True,
                          subset=args.subset)

    X = dataset.images.view(len(dataset), -1).numpy()
    y = dataset.labels.numpy()

    print(f"Total images loaded: {len(dataset)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    train_dataset, test_dataset = split_dataset(dataset, val_ratio=0.2)

    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    X_train, X_test = X[train_indices], X[test_indices]

    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")

    if args.train == 1:
        print("\n=== Training KNN Clustering ===")
        clustering_results = evaluate_clustering(KNNSeparator(), X_train, args.k_clusters)

        print(f"Best K (Elbow): {clustering_results['best_k_elbow']}")
        print(f"Best K (Silhouette): {clustering_results['best_k_silhouette']}")
        print(f"Silhouette Scores: {clustering_results['silhouette_scores']}")

        best_k = clustering_results['best_k_elbow']
        knn_separator = KNNSeparator(k=best_k)
        knn_separator.fit(X_train)

        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoints_dir / f"KNN_Clustering_k{best_k}_{dt_str}.joblib"
        knn_separator.save_model(str(checkpoint_path))
        print(f"KNN Clustering (k={best_k}) saved to {checkpoint_path}")

        # Test clustering on test set
        test_clusters = knn_separator.predict(X_test)
        print(f"Test set cluster distribution: {pd.Series(test_clusters).value_counts().to_dict()}")

    else:
        config_yaml = Path(args.config_yaml)
        if not config_yaml.exists():
            print(f"Config yaml not found: {config_yaml}")
            return

        with open(config_yaml, "r") as f:
            model_paths = yaml.safe_load(f)

        if "KNN_Clustering" not in model_paths:
            print("KNN_Clustering model path not found in config yaml.")
            return

        model_path = model_paths["KNN_Clustering"]
        print(f"\n=== Evaluating KNN Clustering ===")

        knn_separator = KNNSeparator()
        knn_separator.load_model(model_path)

        test_clusters = knn_separator.predict(X_test)
        print(f"Test set cluster distribution: {pd.Series(test_clusters).value_counts().to_dict()}")

    print("\n=== Clustering Complete ===")


if __name__ == "__main__":
    """
    example usage: 
    python scripts/train_eval_KNN.py --data_dirs /path/to/dataset --train 1 --k_clusters 2 4 6 8 10
    """
    main()
