import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pickle
import logging
from tqdm import tqdm
from typing import Union, List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: у KNN нет весов и обучающих данных, там просто хранится весь датасет
# TODO: напиши скрипт для компиляции всего нужного входа в один файл
# TODO: или просто возьми класс из склерн
class KNNPyTorchClassifier:
    """
    Args:
        k (int): по умолчанию 8.
        device (torch.device): Устройство ('cpu' или 'cuda'). Если None, определяется автоматически.
    """
    def __init__(self, k: int = 8, device: Optional[torch.device] = None):
        self.k = k
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"KNNPyTorchClassifier initialized on device: {self.device}")

        # Атрибуты для хранения обучающих данных
        self.X_train: Optional[torch.Tensor] = None # (N_train, D)
        self.y_train: Optional[torch.Tensor] = None # (N_train,)
        self.is_fitted = False
        self.classes_: Optional[torch.Tensor] = None
        self.class_names: Optional[List[str]] = None

    def fit(self, dataloader: DataLoader, class_names: Optional[List[str]] = None):
        """
        Обучает (фиксирует) KNN классификатор на пиксельных значениях изображений из DataLoader.
        Изображения преобразуются в тензоры.

        Args:
            dataloader (DataLoader): DataLoader с обучающими данными (images, labels).
                                     images: torch.Tensor (B, C, H, W)
                                     labels: torch.Tensor (B,)
            class_names (List[str], optional): Список имен классов.
        """
        logger.info("Starting pixel data loading for training...")
        all_images_flat = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Loading Training Data")):
                # data: (B, C, H, W), target: (B,)
                # Преобразуем изображения в векторы: (B, C*H*W)
                images_flat = data.view(data.size(0), -1) # (B, D)
                labels_tensor = target # (B,)

                all_images_flat.append(images_flat)
                all_labels.append(labels_tensor)

        if not all_images_flat:
            raise RuntimeError("Не удалось загрузить данные ни из одного батча.")

        # Объединяем все батчи
        self.X_train = torch.cat(all_images_flat, dim=0).to(self.device) # (N_train, D)
        self.y_train = torch.cat(all_labels, dim=0).to(self.device)     # (N_train,)

        # Сохраняем имена классов, если предоставлены
        if class_names:
            self.class_names = class_names
            self.classes_ = torch.unique(self.y_train)
        else:
            self.class_names = None
            self.classes_ = torch.unique(self.y_train)

        self.is_fitted = True
        logger.info(f"KNNPyTorchClassifier fitted on {self.X_train.size(0)} samples with feature dim {self.X_train.size(1)}.")

    def _predict_batch(self, X_test_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Предсказывает метки и вероятности для батча тестовых изображений.

        Args:
            X_test_batch (torch.Tensor): Тензор изображений (B, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (предсказанные метки (B,), вероятности (B, n_classes)).
        """
        X_test_batch = X_test_batch.to(self.device)
        # X_train уже на self.device

        # Вычисляем расстояния (L2) между батчем тестовых и всеми обучающими
        # dists: (B, N_train)
        # Используем формулу: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        X_train_sq_norms = (self.X_train ** 2).sum(dim=1)  # (N_train,)
        X_test_sq_norms = (X_test_batch ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        cross_term = torch.mm(X_test_batch, self.X_train.t())  # (B, N_train)
        dists_sq = X_train_sq_norms + X_test_sq_norms - 2 * cross_term  # (B, N_train)
        # dists = torch.sqrt(dists_sq) # Не всегда нужно, если сортируем по квадрату

        # Находим k ближайших соседей для каждого тестового примера в батче
        # values: (B, k), indices: (B, k)
        # largest=False -> smallest distances
        topk_distances_sq, topk_indices = torch.topk(dists_sq, self.k, dim=1, largest=False, sorted=True)

        # Получаем метки k соседей (B, k)
        neighbors_labels = self.y_train[topk_indices]  # (B, k)

        #Определяем предсказания
        batch_size = X_test_batch.size(0)
        predictions = torch.zeros(batch_size, dtype=self.y_train.dtype, device=self.device)
        probabilities = torch.zeros(batch_size, len(self.classes_), device=self.device)

        for i in range(batch_size):
            neighbor_labels_i = neighbors_labels[i] # (k,)
            # Подсчитываем количество для каждого класса
            unique_labels, counts = torch.unique(neighbor_labels_i, return_counts=True)
            # Находим индекс класса с максимальным количеством
            max_count_idx = torch.argmax(counts)
            predicted_label = unique_labels[max_count_idx]
            predictions[i] = predicted_label

            # Вычисляем вероятности 
            total_votes = neighbor_labels_i.numel()
            for label, count in zip(unique_labels, counts):
                class_idx = (self.classes_ == label).nonzero(as_tuple=True)[0]
                if class_idx.numel() > 0:
                    probabilities[i, class_idx] = count.float() / total_votes

        return predictions, probabilities

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Предсказывает метки классов для данных из DataLoader.

        Args:
            dataloader (DataLoader): DataLoader с данными для предсказания.

        Returns:
            np.ndarray: Массив предсказанных меток размером (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("KNNPyTorchClassifier is not fitted yet. Call 'fit' first.")

        logger.info("Starting prediction...")
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Predicting")):
                #  (B, C, H, W)
                # Преобразуем изображения в 1D векторы: (B, D)
                X_test_batch_flat = data.view(data.size(0), -1) # (B, D)

                try:
                    batch_predictions, _ = self._predict_batch(X_test_batch_flat) # (B,)
                    all_predictions.append(batch_predictions.cpu().numpy())
                except Exception as e:
                    logger.error(f"Ошибка при обработке батча {batch_idx} для предсказания: {e}")
                    continue # Пропустить батч с ошибкой

        if not all_predictions:
            raise RuntimeError("Не удалось получить предсказания ни для одного батча.")

        predictions = np.concatenate(all_predictions, axis=0) # (N_predict,)
        logger.info("Prediction completed.")
        return predictions

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """
        Предсказывает вероятности классов для данных из DataLoader.

        Args:
            dataloader (DataLoader): DataLoader с данными для предсказания.

        Returns:
            np.ndarray: Массив вероятностей размером (N, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("KNNPyTorchClassifier is not fitted yet. Call 'fit' first.")

        logger.info("Starting prediction (probabilities)...")
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Predicting (Probabilities)")):
                #  (B, C, H, W)
                # Преобразуем изображения в векторы: (B, D)
                X_test_batch_flat = data.view(data.size(0), -1) # (B, D)

                try:
                    _, batch_probabilities = self._predict_batch(X_test_batch_flat) # (B, n_classes)
                    all_probabilities.append(batch_probabilities.cpu().numpy())
                except Exception as e:
                    logger.error(f"Ошибка при обработке батча {batch_idx} для предсказания вероятностей: {e}")
                    continue # Пропустить батч с ошибкой

        if not all_probabilities:
            raise RuntimeError("Не удалось получить вероятности ни для одного батча.")

        probabilities = np.concatenate(all_probabilities, axis=0) # (N_predict, n_classes)
        logger.info("Prediction (probabilities) completed.")
        return probabilities

    def save_weights(self, path: Union[str, Path]):
        """
        Сохраняет "веса" KNN (обучающие данные) в файл.

        Args:
            path (Union[str, Path]): Путь к файлу для сохранения.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save weights: KNN is not fitted.")

        weights_data = {
            'X_train': self.X_train.cpu(), # Сохраняем на CPU
            'y_train': self.y_train.cpu(),
            'k': self.k,
            'device': str(self.device), # Сохраняем строку устройства
            'classes_': self.classes_.cpu() if self.classes_ is not None else None,
            'class_names': self.class_names,
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(weights_data, f)
            logger.info(f"KNNPyTorchClassifier weights saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save KNNPyTorchClassifier weights: {e}")
            raise

    def load_weights(self, path: Union[str, Path], device: Optional[torch.device] = None):
        """
        Загружает обученный KNN классификатор из файла.

        Args:
            path (Union[str, Path]): Путь к файлу с сохраненной моделью.
            device (torch.device, optional): Устройство для загрузки. Если None, определяется автоматически.

        Returns:
            KNNPyTorchClassifier: Загруженный экземпляр классификатора.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            with open(path, 'rb') as f:
                weights_data = pickle.load(f)

            # Создаем новый экземпляр классификатора
            # Учитываем, что устройство может быть другим при загрузке

            # Загружаем данные
            self.X_train = weights_data['X_train'].to(self.device) # Переносим на нужное устройство
            self.y_train = weights_data['y_train'].to(self.device)
            self.classes_ = weights_data['classes_'].to(self.device) if weights_data['classes_'] is not None else None
            self.class_names = weights_data.get('class_names')

            self.is_fitted = True

            logger.info(f"KNNPyTorchClassifier model loaded from {path} to device {self.device}")
        except Exception as e:
            logger.error(f"Failed to load KNNPyTorchClassifier model: {e}")
            raise

    def evaluate(self, dataloader: DataLoader, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """

        Args:
            dataloader (DataLoader): DataLoader с данными для оценки (images, true_labels).
            class_names (List[str], optional): Список имен классов для отчета.

        Returns:
            dict: Словарь с метриками, например, {'accuracy': float, 'report': str, 'confusion_matrix': np.ndarray}.
        """
        # --- Вычисление метрик на PyTorch ---
        # Используем torch для вычисления accuracy, confusion matrix, precision, recall, f1
        # Это заменяет sklearn.metrics

        if not self.is_fitted:
            raise RuntimeError("KNNPyTorchClassifier is not fitted yet. Call 'fit' first.")

        logger.info("Starting evaluation...")
        all_true_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
                #  (B, C, H, W), target: (B,)
                # Преобразуем изображения в 1D векторы: (B, D)
                X_test_batch_flat = data.view(data.size(0), -1) # (B, D)

                try:
                    batch_predictions, _ = self._predict_batch(X_test_batch_flat) # (B,)

                    all_predictions.append(batch_predictions)
                    all_true_labels.append(target)
                except Exception as e:
                    logger.error(f"Ошибка при обработке батча {batch_idx} для оценки: {e}")
                    continue # Пропустить батч с ошибкой

        if not all_true_labels or not all_predictions:
            raise RuntimeError("Не удалось получить предсказания для оценки.")

        # Объединяем все батчи в тензоры PyTorch
        y_true_tensor = torch.cat(all_true_labels, dim=0).to(self.device)
        y_pred_tensor = torch.cat(all_predictions, dim=0).to(self.device)

        # Вычисление метрик
        # Accuracy
        correct = (y_true_tensor == y_pred_tensor).sum().item()
        total = y_true_tensor.size(0)
        accuracy = correct / total if total > 0 else 0.0

        #Confusion Matrix

        n_classes = len(self.classes_)
        cm_indices = y_true_tensor * n_classes + y_pred_tensor
        cm_counts = torch.bincount(cm_indices, minlength=n_classes * n_classes)
        cm = cm_counts.view(n_classes, n_classes).cpu().numpy()

        # Precision, Recall, F1 для каждого класса
        # Вычисляем из confusion matrix
        precision_per_class = np.zeros(n_classes)
        recall_per_class = np.zeros(n_classes)
        f1_per_class = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]  # True Positives
            fp = cm[:, i].sum() - tp  # False Positives
            fn = cm[i, :].sum() - tp  # False Negatives

            precision_per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) > 0 else 0.0

        macro_precision = precision_per_class.mean()
        macro_recall = recall_per_class.mean()
        macro_f1 = f1_per_class.mean()

        total_tp = cm.diagonal().sum()
        total_actual = y_true_tensor.size(0)
        total_pred = y_pred_tensor.size(0)

        micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
        micro_recall = total_tp / total_actual if total_actual > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0


        report_lines = [
            f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
            f"{'-'*40}"
        ]
        class_labels = self.class_names if self.class_names else [f"Class_{i}" for i in range(n_classes)]
        for i in range(n_classes):
            line = f"{class_labels[i]:<10} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f}"
            report_lines.append(line)
        report_lines.append(f"{'-'*40}")
        report_lines.append(f"{'Micro Avg':<10} {micro_precision:<10.4f} {micro_recall:<10.4f} {micro_f1:<10.4f}")
        report_lines.append(f"{'Macro Avg':<10} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        report_str = "\n".join(report_lines)

        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return {
            'accuracy': accuracy,
            'classification_report': report_str,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
        }
