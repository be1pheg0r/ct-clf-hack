from torchvision.models.resnet import resnet50
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim, amp
import torch
from tqdm import tqdm
from numpy import ndarray
import warnings
from pathlib import Path
from typing import Optional


warnings.filterwarnings("ignore", category=FutureWarning)


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


class ModelCNN(nn.Module):
    INPUT_SIZE = (224, 224)

    def __init__(self, n_models: int = 1, num_classes: int = 2, device: Optional[str] = None):
        super().__init__()

        self.n_models = n_models
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.transforms = Compose([
            Resize(ModelCNN.INPUT_SIZE),
            ToTensor(),
            Normalize(mean=[0.485], std=[0.229])
        ])

        self.models = nn.ModuleList([self.init_model() for _ in range(n_models)])
        self.criterions = [nn.CrossEntropyLoss().to(self.device) for _ in range(n_models)]
        self.optimizers = [
            optim.Adam(self.models[i].parameters(), lr=5e-4) for i in range(n_models)
        ]
        self.schedulers = [
            optim.lr_scheduler.StepLR(self.optimizers[i], step_size=5, gamma=0.1) for i in range(n_models)
        ]

    def check_model_index(self, n: int):
        if n >= self.n_models:
            raise ValueError('There is no n-th model.')

    def change_first_layer(self, model):
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=model.conv1.bias is not None)
        return model

    def init_model(self):
        model = resnet50(weights=None)
        model = self.change_first_layer(model)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        return model.to(self.device)

    def forward(self, x, n: int = 0):
        self.check_model_index(n)

        return self.models[n](x)

    def load_data_to_dataloader(self, images: list[ndarray], labels: list[int], batch_size: int = 32):
        dataset = CTDataset(images, labels, transform=self.transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_model(self, dataloader, n: int = 0, epochs: int = 10):
        self.check_model_index(n)

        model = self.models[n]
        optimizer = self.optimizers[n]
        scheduler = self.schedulers[n]
        criterion = self.criterions[n]

        use_amp = self.device == "cuda"
        scaler = amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            loop = tqdm(dataloader, desc=f"Model {n} | Epoch {epoch + 1}/{epochs}")
            for images, labels in loop:
                images, labels = images.to(self.device).float(), labels.to(self.device)

                optimizer.zero_grad()
                if use_amp:
                    with amp.autocast(device_type=self.device):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_postfix(loss=loss.item(), acc=correct / total)

            scheduler.step()
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = correct / total
            print(f"Model {n} | Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    def save(self, n: int = 0, dpath: Optional[str] = None, all: bool = False):

        def save_model(model, i: int):
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            save_path = Path(dpath) / f"resnet50_model{i}.pth" if dpath else Path(f"resnet50_model{i}.pth")
            torch.save(state, save_path)
            print(f"Model {i} saved to {save_path}")
        
        self.check_model_index(n)

        if all:
            for i in range(self.n_models):
                model = self.models[i]
                save_model(model, i)
        else:
            model = self.models[n]
            save_model(model, n)


    def load(self, model_path: str, n: int = 0):
        self.check_model_index(n)

        model = self.models[n]
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(str(model_path), map_location=self.device)

        try:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            raise

        model.eval()
        print(f"Model {n} loaded from {model_path}")

    def predict(self, images: list[ndarray], batch_size: int = 32, n: int = 0) -> list[int]:
        self.check_model_index(n)
        
        dataset = CTDataset(images, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = self.models[n]
        model.eval()

        preds_all = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Predicting model {n}"):
                batch = batch.to(self.device).float()
                outputs = model(batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds_all.extend(preds)
        return preds_all
