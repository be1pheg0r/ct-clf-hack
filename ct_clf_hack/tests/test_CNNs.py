from os import walk
import pydicom as dicom
import numpy as np
from PIL import Image

from ct_clf_hack.models.ModelCNN import ModelCNN

images = []
labels = []

for _, dirnames, _ in walk('../../data/'):
    for dir in dirnames:
        if dir == "pneumothorax_anon":
            continue
        dir_path = f'../../data/{dir}'
        for dirpath, dirnames, filenames in walk(dir_path):
            for file_name in filenames:
                if '.' in file_name:
                    continue
                file = f'{dirpath}/{file_name}'
                try:
                    img_array = dicom.dcmread(file).pixel_array
                    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
                    img_array = (img_array * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array.squeeze()).convert("L")
                    images.append(pil_img)
                    if dir == 'norma_anon':
                        labels.append(0)
                    else:
                        labels.append(1)
                except Exception as e:
                    print(f"Ошибка с файлом {file}: {e}")

model = ModelCNN('ResNet50', num_classes=2)
print(f"Using model architecture: {model.model_name}")
print(f"Number of classes: {model.num_classes}")
print(f"Using device: {model.device}")
print(f"Total images loaded: {len(images)}")
if len(images) != len(labels):
    raise ValueError("The number of images must be equal to the number of labels.")
model.load_data(images, labels)
model.train()
