from torchvision.models.densenet import densenet121
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50
from torchvision.models.convnext import convnext_tiny

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim, cuda, amp, max
from torch import save as model_save
import torch

from tqdm import tqdm
from numpy import ndarray

import warnings
from pathlib import Path

from typing import Optional

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

    def __init__(self, model_name: str, num_classes: int = 2, device: Optional[str] = None):
        if model_name not in ModelCNN.__ARCHITECTURE:
            raise Exception('There is no such model architecture.')

        self.model_name = model_name
        self.num_classes = num_classes

        if device is None:
            self.device = "cuda" if cuda.is_available() else "cpu"
        else:
            self.device = device

        self.transforms = Compose([
            Resize(ModelCNN.__INPUT_SIZE[self.model_name]),
            ToTensor(),
            Normalize(mean=[0.485], std=[0.229])
        ])
        self.init_model()

    def change_first_layer(self, model):
        if self.model_name == "ResNet50":
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride, padding=model.conv1.padding,
                                    bias=model.conv1.bias is not None)
        elif self.model_name == "DenseNet121":
            model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels,
                                             kernel_size=model.features.conv0.kernel_size,
                                             stride=model.features.conv0.stride, padding=model.features.conv0.padding,
                                             bias=model.features.conv0.bias is not None)
        elif self.model_name == "Inception_V3":
            model.Conv2d_1a_3x3.conv = nn.Conv2d(1, model.Conv2d_1a_3x3.conv.out_channels,
                                                 kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
                                                 stride=model.Conv2d_1a_3x3.conv.stride,
                                                 padding=model.Conv2d_1a_3x3.conv.padding,
                                                 bias=model.Conv2d_1a_3x3.conv.bias is not None)
        elif self.model_name == "ConvNeXt_Tiny":
            original_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
        return model

    def init_model(self):
        architecture = ModelCNN.__ARCHITECTURE[self.model_name]
        self.model = architecture(weights=None)
        self.model = self.change_first_layer(self.model)

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

    def load_data(self, loader: DataLoader):
        self.dataloader = loader

    def train(self, epochs: int = 10):
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
                if images.ndim == 3:
                    images = images.unsqueeze(1)
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
                _, predicted = max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item(), acc=correct / total)

            self.scheduler.step()

            epoch_loss = running_loss / len(self.dataloader.dataset)
            epoch_acc = correct / total

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    def save(self, dpath: str):
        """Save model to specified path"""
        model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                   nn.DataParallel) else self.model.state_dict()
        model_save(model_state, dpath)

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

    def predict(self, dataset: Dataset, batch_size: int = 32) -> list[int]:
        """Predict using dataset"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch, _ in tqdm(dataloader, desc="Predicting"):
                batch = batch.to(self.device).float()
                outputs = self.model(batch)
                if self.model_name == "Inception_V3" and self.model.training:
                    outputs = outputs.logits
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
        return all_preds

