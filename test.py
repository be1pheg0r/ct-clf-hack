from os import walk
import pydicom as dicom
import numpy as np
from PIL import Image

from ModelCNN import ModelCNN


images = []
labels = []
for _, dirnames, _ in walk('data'):
	for dir in dirnames:
		for dirpath, dirnames, filenames in walk('data/' + dir):
			for file_name in filenames:
				file = dirpath + '/' + file_name
				pneumothorax = 0
				if dir == 'pneumotorax_anon':
					pneumothorax = np.array([dicom.dcmread(file).pixel_array ])[0]
					for pneumo in pneumothorax:
						image = pneumo
						image = (image - np.min(image)) / (np.max(image) - np.min(image))
						image = (image * 255).astype(np.uint8)
						try:
							rgb_image = Image.fromarray(image.squeeze()).convert("RGB")
						except:
							print(file)
				else:
					image = np.array(dicom.dcmread(file).pixel_array)
					image = (image - np.min(image)) / (np.max(image) - np.min(image))
					image = (image * 255).astype(np.uint8)
					rgb_image = Image.fromarray(image.squeeze()).convert("RGB")

				images.append(rgb_image)
			if dir == 'norma_anon':
				labels += [0] * len(filenames)
			else:
				labels += [1] * len(filenames) if len(filenames) > 1 else [1] * len(pneumothorax)
				
model = ModelCNN('ResNet50')
model.load_data(images, labels)
model.train()
