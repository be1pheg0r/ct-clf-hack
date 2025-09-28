import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class KMeansTorch:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.centroids = None

    def _gather_from_loader(self, loader):
        """Собираем все данные из DataLoader в один тензор"""
        all_batches = []
        for batch in loader:
            # если батч — (X, y), то берём только X
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            all_batches.append(X)
        return torch.cat(all_batches, dim=0)

    def fit(self, data):
        """Обучение на DataLoader или на Tensor"""
        if hasattr(data, "__iter__") and not isinstance(data, torch.Tensor):
            X = self._gather_from_loader(data)
        else:
            X = data
        X = X.to(self.device)

        N, D = X.shape
        indices = torch.randperm(N)[:self.n_clusters]
        self.centroids = X[indices]

        for i in range(self.max_iter):
            print('iter:', i)
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(dim=0))
                else:
                    new_centroids.append(X[torch.randint(0, N, (1,))])
            new_centroids = torch.stack(new_centroids)

            shift = torch.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        return labels.cpu().numpy()

    def predict(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, torch.Tensor):
            X = self._gather_from_loader(data)
        else:
            X = data
        X = X.to(self.device)

        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1).cpu().numpy()

    def inertia(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, torch.Tensor):
            X = self._gather_from_loader(data)
        else:
            X = data
        X = X.to(self.device)

        labels = self.predict(X)
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += ((cluster_points - self.centroids[k])**2).sum().item()
        return inertia

    def evaluate(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, torch.Tensor):
            X = self._gather_from_loader(data)
        else:
            X = data
        labels = self.predict(X)

        inertia_val = self.inertia(X)
        metrics = {"inertia": inertia_val}

        X_np = X.cpu().numpy()
        labels_np = labels

        if len(set(labels_np)) > 1:
            metrics["silhouette"] = silhouette_score(X_np, labels_np)
            metrics["calinski_harabasz"] = calinski_harabasz_score(X_np, labels_np)
            metrics["davies_bouldin"] = davies_bouldin_score(X_np, labels_np)
        else:
            metrics["silhouette"] = None
            metrics["calinski_harabasz"] = None
            metrics["davies_bouldin"] = None

        return metrics
    
    def save(self, path: str):
        """Сохраняем модель на диск"""
        torch.save({
			"n_clusters": self.n_clusters,
			"max_iter": self.max_iter,
			"tol": self.tol,
			"device": self.device,
			"centroids": self.centroids,
		}, path)

    def load_weights(self, path: str):
        """Загружаем модель с диска"""
        checkpoint = torch.load(path, map_location=self.device)
        self.n_clusters=checkpoint["n_clusters"],
        self.max_iter=checkpoint["max_iter"],
        self.tol=checkpoint["tol"],
        self.centroids = checkpoint["centroids"].to(self.device)