# Importing modules and packages
import random
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

import nanotorch as torch
import nanotorch.nn as nn

# Setting seed for reproducibility
np.random.seed(1337)
random.seed(1337)

# Loading dataset from Sklearn
X, y = make_moons(100, noise=0.1)
y = y * 2 - 1


# Build classification model using NanoTorch's APIs
class Net(nn.Module):

    def __init__(self, in_features, out_features, hidden_features) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, hidden_features)
        self.linear_3 = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        return x


# Defining hyperparameters
in_features = 2
out_features = 1
hidden_features = 16
learning_rate = 0.01
alpha = 1e-4
momentum = 0.9
epochs = 80

# Load defined model
model = Net(in_features, out_features, hidden_features)
print(model)
print(f"Number of parameters: {len(model.parameters())}")

# Load loss function and optimizer
criterion = nn.HingeEmbeddingLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)

# Model training loop
for epoch in range(epochs):
    # Forward pass
    inputs = [list(map(torch.tensor, x)) for x in X]
    y_pred = list(map(model, inputs))
    loss = criterion(y, y_pred)

    # L2 Regularization
    loss += alpha * sum(
        (parameter * parameter for parameter in model.parameters())
    )

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update using gradient descent
    optimizer.step()

    # Calculating accuracy
    accuracy = [(yi > 0) == (y_i.item() > 0) for yi, y_i in zip(y, y_pred)]
    accuracy = sum(accuracy) / len(accuracy)

    if epoch % 1 == 0:
        print(
            f"Epoch: [{epoch:>02}/{epochs}] | Loss: {loss.item():.4f} "
            f"| Accuracy: {accuracy:.2f}"
        )

# Plotting margin
step = 0.25

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
)
Xmesh = np.c_[xx.ravel(), yy.ravel()]

inputs = [list(map(torch.tensor, x)) for x in Xmesh]
y_pred = list(map(model, inputs))

Z = np.array([y_i.data > 0 for y_i in y_pred])
Z = Z.reshape(xx.shape)

_, axes = plt.subplots(1, 2)

axes[0].scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="gnuplot", edgecolor="k")
axes[1].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

plt.show()
