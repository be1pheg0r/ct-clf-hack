"""
Модуль для создания, обучения, сохранения и загрузки сверточных нейронных сетей.
Архитектуры: ResNet50, DenseNet121, Inception_V3, ConvNeXt_Tiny.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torch import nn, optim, cuda, amp
from torch import save as model_save
from torch.utils.data import DataLoader
from torchvision.models.convnext import convnext_tiny
from torchvision.models.densenet import densenet121
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from tqdm import tqdm

from ct_clf_hack.shared.data_utils import CTDataset, convert_to_rgb

warnings.filterwarnings("ignore", category=FutureWarning)

class ModelCNN:
    __INPUT_SIZE = {
        "ResNet50": (224, 224),
        "DenseNet121": (224, 224),
        "Inception_V3": (299, 299),
        "ConvNeXt_Tiny": (224, 224)
    }
    __ARCHITECTURE = {
        "ResNet50": resnet50,
        "DenseNet121": densenet121,
        "Inception_V3": inception_v3,
        "ConvNeXt_Tiny": convnext_tiny
    }

    def __init__(self, model_name: str, num_classes: int = 2, device: Optional[torch.device] = None):
        """
        CNN модель с возможностью обучения, сохранения и загрузки.
        Args:
            model_name (str): Название архитектуры модели. Поддерживаемые: "ResNet50", "DenseNet121", "Inception_V3", "ConvNeXt_Tiny".
            num_classes (int): Количество классов для классификации.
            device (torch.device, optional): Устройство для вычислений (CPU или GPU). По умолчанию выбирается автоматически.
        """
        if model_name not in ModelCNN.__ARCHITECTURE:
            raise Exception('There is no such model architecture.')

        self.model_name = model_name
        self.num_classes = num_classes

        self.device = device

        self.transforms = Compose([
            Resize(ModelCNN.__INPUT_SIZE[self.model_name]),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.init_model()

    def init_model(self):
        architecture = ModelCNN.__ARCHITECTURE[self.model_name]
        self.model = architecture(weights='IMAGENET1K_V1')

        if self.model_name == "ResNet50":
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "DenseNet121":
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "Inception_V3":
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "ConvNeXt_Tiny":
            num_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_features, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        if self.device == "cuda" and cuda.device_count() > 1:
            print(f"Using {cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        self.criterion.to(self.device)

    def load_data(self, images: list[ndarray], labels: list[int]):
        """
        Загружает данные и создает DataLoader (только для обучения).
        Args:
            images (list[ndarray]): Список изображений.
            labels (list[int]): Список меток классов.
        """
        rgb_images = [convert_to_rgb(img) for img in images]

        dataset = CTDataset(rgb_images, labels, transform=self.transforms)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataset

    def train(self, epochs: int = 10):
        """
        Обучает модель на загруженных данных.
        Args:
            epochs (int): Количество эпох для обучения.
        """
        use_amp = self.device == "cuda"
        scaler = amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, labels in loop:
                images, labels = images.to(self.device).float(), labels.to(self.device)

                self.optimizer.zero_grad()

                if use_amp:
                    with amp.autocast(device_type=self.device):
                        outputs = self.model(images)
                        if self.model_name == "Inception_V3" and self.model.training:
                            outputs = outputs.logits
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    if self.model_name == "Inception_V3" and self.model.training:
                        outputs = outputs.logits
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item(), acc=correct / total)

            self.scheduler.step()

            epoch_loss = running_loss / len(self.dataloader.dataset)
            epoch_acc = correct / total

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    def save(self, spath: Optional[str] = None):
        """
        Сохраняет модель на диск.
        Args:
            spath (str, optional): Путь для сохранения модели. Если None, сохраняется в 'trained_<model_name>.pth'.
        """
        model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                   nn.DataParallel) else self.model.state_dict()
        if spath:
            spath = Path(spath)
            model_save(model_state, spath)
        else:
            save_path = f'trained_{self.model_name}.pth'
            model_save(model_state, save_path)

    def load(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(str(model_path), map_location=self.device)

        num_classes = None
        if self.model_name == "ResNet50":
            num_classes = state_dict["fc.weight"].shape[0]
        elif self.model_name == "DenseNet121":
            num_classes = state_dict["classifier.weight"].shape[0]
        elif self.model_name == "Inception_V3":
            num_classes = state_dict["fc.weight"].shape[0]
        elif self.model_name == "ConvNeXt_Tiny":
            num_classes = state_dict["classifier.2.weight"].shape[0]

        if num_classes is not None and num_classes != self.num_classes:
            self.num_classes = num_classes
            self.init_model()

        try:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Trying to reinitialize model with correct num_classes...")
            self.init_model()
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict_proba(self, images: list[ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Предсказывает вероятности классов для заданных изображений.
        Args:
            images (list[ndarray]): Список изображений.
            batch_size (int): Размер батча для предсказания.
        Returns:
            np.ndarray: Массив вероятностей классов.
        """
        rgb_images = [convert_to_rgb(img) for img in images]
        if isinstance(rgb_images[0], np.ndarray):
            rgb_images = [Image.fromarray(img.astype(np.uint8)) for img in rgb_images]

        dataset = CTDataset(rgb_images, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = batch.to(self.device).float()
                outputs = self.model(batch)
                if self.model_name == "Inception_V3" and self.model.training:
                    outputs = outputs.logits
                all_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        return np.array(all_preds)

    def predict(self, images: list[ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Предсказывает классы для заданных изображений.
        Args:
            images (list[ndarray]): Список изображений.
            batch_size (int): Размер батча для предсказания.
        Returns:
            np.ndarray: Массив предсказанных классов.
        """
        proba = self.predict_proba(images, batch_size)
        return np.argmax(proba, axis=1)


class ConvSensus:
    def __init__(self, models_config: dict, device: Optional[torch.device] = None):
        """
         Ансамбль из нескольких моделей CNN для улучшения качества предсказаний.
         Args:
             models_config (dict): Конфигурация моделей с архитектурами и путями к
                                     сохраненным весам.
             device (torch.device, optional): Устройство для вычислений (CPU или GPU). По умолчанию выбирается автоматически.
         """
        self.models = []
        for arch, cfg in models_config.items():
            checkpoints_path = cfg.get("local", None)
            if not checkpoints_path or not Path(checkpoints_path).exists():
                print(f"Warning: Checkpoint for {arch} not found at {checkpoints_path}, skipping.")
                continue
            model = ModelCNN(arch, device=device)
            model.load(checkpoints_path)
            self.models.append(model)

    def predict_proba(self, images: list[ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Предсказывает усредненные вероятности классов от всех моделей ансамбля.
        Args:
            images (list[ndarray]): Список изображений.
            batch_size (int): Размер батча для предсказания.
        Returns:
            np.ndarray: Массив усредненных вероятностей классов.
        """
        if not self.models:
            raise Exception("No models loaded in the ensemble.")

        all_probas = []
        for model in self.models:
            proba = model.predict_proba(images, batch_size)
            all_probas.append(proba)

        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba

    def predict(self, images: list[ndarray], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает классы и вероятности для заданных изображений.
        Args:
            images (list[ndarray]): Список изображений.
            batch_size (int): Размер батча для предсказания.
        Returns:
            tuple[np.ndarray, np.ndarray]: Кортеж из массива вероятностей и массива предсказанных классов.
        """
        proba = self.predict_proba(images, batch_size)
        return proba, np.argmax(proba, axis=1)


def check_checkpoints(models_cfg: dict) -> bool:
    """
    Проверяет наличие всех контрольных точек моделей.
    Args:
        models_cfg (dict): Конфигурация моделей с архитектурами и путями к
                             сохраненным весам.
    Returns:
        bool: True, если все контрольные точки существуют, иначе False.
    """
    all_exist = True
    for arch, cfg in models_cfg.items():
        checkpoints_path = cfg.get("local", None)
        if not checkpoints_path or not Path(checkpoints_path).exists():
            print(f"Warning: Checkpoint for {arch} not found at {checkpoints_path}.")
            all_exist = False
    return all_exist