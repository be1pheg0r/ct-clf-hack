import numpy as np
import pandas as pd
import torch
from typing import Union, Optional
from sklearn.cluster import KMeans
import joblib


class KNNSeparator:
    def __init__(self, data: Optional[Union[np.ndarray, torch.Tensor]] = None, k: int = 8) -> None:
        self.k = k
        self.model = KMeans(n_clusters=self.k)
        if data is not None:
            self.load_data(data)

    def load_data(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        self.fit(data)

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        self.model.fit(data)

    def predict(self, vec: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        return self.model.predict(vec)

    def fit_predict(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return self.model.fit_predict(data)

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load_model(self, path: str, inplace: bool = True) -> Optional['KNNSeparator']:
        loaded_model = joblib.load(path)
        self.model = loaded_model
        return self if inplace else None

    def elbow_method(self, data: Union[np.ndarray, torch.Tensor], max_k: int = 10) -> tuple[int, pd.DataFrame]:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        sse = []
        k_values = list(range(1, max_k + 1))

        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        best_k = k_values[np.argmin(np.diff(sse, 2)) + 1] if len(sse) > 2 else 1
        results_df = pd.DataFrame({'k': k_values, 'sse': sse})
        return best_k, results_df

