import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from DataManager import DataManager
from KNN import KNNPyTorchClassifier
from KMeans import KMeansTorch


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    PATH = 'data'
    dataset = DataManager.create_dataset_from_dirs(PATH)
    dataloader = DataManager.create_dataloader(dataset, batch_size=32)
    
    N = 8
    k_means = KMeansTorch(N)

    # обучение
    k_means.fit(dataset)

    # предсказание
    '''
    path = ''
    k_means.load_weights(path)
    '''

    pred = k_means.predict(dataloader)

    cluster_datasets = []
    for c in np.unique(pred):
        indices = np.where(pred == c)[0]
        cluster_subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(cluster_subset, batch_size=16, shuffle=True)
        cluster_datasets.append(loader)

    # предсказание
    '''
    predictions = []
    '''

    for i in range(N):
        cluster = cluster_datasets[i]
        knn = KNNPyTorchClassifier(N)
        # обучение
        knn.fit(cluster)
        knn.save_weights(f'knn_model_cluster_{i}.pkl')

        # предсказание
        '''
        path = ''
        knn.load_weights()
        pred = knn.predict_proba(cluster)
        predictions.append(pred)
        '''
    
    # предсказание
    '''
    predictions = np.array(predictions)
    mean_probabilities = predictions.mean(axis=0)
    max_idx = np.argmax(mean_probabilities)
    print(DataManager.get_class_name(max_idx))
    '''