import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from mlflow.tracking import MlflowClient

# 1. Инициализация эксперимента
mlflow.set_experiment("doc-classifier")


# Генерируем синтетические данные
def get_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    return train_loader, X_test, y_test


# Простая нейросеть
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)


# 2. Функция обучения
def train_once(lr, epochs):
    with mlflow.start_run() as run:
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        train_loader, X_te, y_te = get_data()
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Цикл обучения
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        # Оценка
        model.eval()
        with torch.no_grad():
            y_pred = model(X_te).argmax(dim=1).numpy()
            acc = accuracy_score(y_te.numpy(), y_pred)

        mlflow.log_metric("test_accuracy", acc)

        # Сохраняем модель в формате pickle (обходит требование TensorSpec для pt2)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

        # Дополнительный артефакт (график)
        plt.figure()
        plt.plot([1, 2, 3], [acc, acc, acc])
        plt.title(f"Accuracy: {acc:.2f}")
        plt.savefig("accuracy_plot.png")
        mlflow.log_artifact("accuracy_plot.png")

        if os.path.exists("accuracy_plot.png"):
            os.remove("accuracy_plot.png")

        # Добавляем тег
        mlflow.set_tag("model_type", "SimpleNet")

        print(f"Запуск: lr={lr}, epochs={epochs}, acc={acc:.4f}")
        return acc, run.info.run_id


# 3. Запуск экспериментов
if __name__ == "__main__":
    best_acc = 0
    best_run_id = None

    for lr in [0.0001, 0.001, 0.01]:
        for epochs in [5, 10, 20]:
            acc, run_id = train_once(lr, epochs)
            if acc > best_acc:
                best_acc = acc
                best_run_id = run_id

    print(f"\nЛучшая точность: {best_acc}")
    print(f"Лучший run_id: {best_run_id}")

    # 4. Регистрация модели и перевод в Staging (через MlflowClient)
    if best_run_id:
        client = MlflowClient()
        model_uri = f"runs:/{best_run_id}/model"

        # Регистрируем модель
        model_version = mlflow.register_model(model_uri, "DocClassifierBest")
        print(f"Модель зарегистрирована. Версия: {model_version.version}")

        # Переводим в стадию Staging
        client.transition_model_version_stage(
            name="DocClassifierBest",
            version=model_version.version,
            stage="Staging"
        )
        print(f"Модель версии {model_version.version} переведена в стадию Staging")