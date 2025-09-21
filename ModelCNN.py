from torchvision.models.densenet import densenet121
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50
from torchvision.models.efficientnet import efficientnet_b3
from torchvision.models.convnext import convnext_tiny
from torchvision.models import (DenseNet121_Weights, Inception_V3_Weights, ResNet50_Weights,
                                EfficientNet_B3_Weights, ConvNeXt_Tiny_Weights)

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim, cuda, amp, max
from torch import save as model_save

from tqdm import tqdm
from numpy import ndarray


class CTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


class ModelCNN:
    __INPUT_SIZE = {"ResNet50": (224, 224),
                    "DenseNet121": (224, 224),
                    "Inception_V3": (299, 299),
                    "EfficientNet_B3": (300, 300),
                    "ConvNeXt_Tiny": (224, 224)}
    __WEIGHTS = {"ResNet50": ResNet50_Weights.IMAGENET1K_V1,
                 "DenseNet121": DenseNet121_Weights.IMAGENET1K_V1,
                 "Inception_V3": Inception_V3_Weights.IMAGENET1K_V1,
                 "EfficientNet_B3": EfficientNet_B3_Weights.IMAGENET1K_V1,
                 "ConvNeXt_Tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1}
    __ARCHITECTURE = {"ResNet50": resnet50,
                      "DenseNet121": densenet121,
                      "Inception_V3": inception_v3,
                      "EfficientNet_B3": efficientnet_b3,
                      "ConvNeXt_Tiny": convnext_tiny}


    def __init__(self, model_name: str):
        if model_name not in ModelCNN.__ARCHITECTURE:
            raise Exception('There is no such model architecture.')
        
        self.__model_name = model_name
        self.__transforms = Compose([Resize(ModelCNN.__INPUT_SIZE[self.__model_name]),
                                     ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])])
        
        self.__init_model()
        
    def __init_model(self):
        architecture = ModelCNN.__ARCHITECTURE[self.__model_name]
        weights = ModelCNN.__WEIGHTS[self.__model_name]
        self.__model = architecture(weights=weights)
        num_features = self.__model.fc.in_features
        self.__model.fc = nn.Linear(num_features, 2)

        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=1e-4)
        self.__scheduler = optim.lr_scheduler.StepLR(self.__optimizer, step_size=5, gamma=0.1)

        self.__device = "cuda" if cuda.is_available() else "cpu"

        if cuda.device_count() > 1:
            print(f"Using {cuda.device_count()} GPUs!")
            self.__model = nn.DataParallel(self.__model)

        self.__model = self.__model.to(self.__device)
        self.__criterion.to(self.__device)
        
    def load_data(self, images: list[ndarray], labels: list[int]):
        dataset = CTDataset(images, labels, transform=self.__transforms)
        self.__dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self, epochs: int=10):
        scaler = amp.GradScaler()

        for epoch in range(epochs):
            self.__model.train()
            running_loss = 0.0
            
            correct = 0
            total = 0

            loop = tqdm(self.__dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in loop:
                images, labels = images.to(self.__device).float(), labels.to(self.__device)

                self.__optimizer.zero_grad()

                with amp.autocast(self.__device):
                    outputs = self.__model(images)
                    loss = self.__criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(self.__optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                _, predicted = max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item(), acc=correct/total)

            self.__scheduler.step()

            epoch_loss = running_loss / len(self.__dataloader.dataset)
            epoch_acc = correct / total

            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    def save(self):
        model_save(self.__model.state_dict(), f'trained_{self.__model_name}.pth')
