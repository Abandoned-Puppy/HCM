import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
import numpy as np
from scipy.spatial import cKDTree


# --------------------
# 0. Random seed & device
# --------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------
# 1. Dataset utilities
# --------------------
def to_one_hot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    labels = labels.long()
    return torch.eye(num_classes, device=labels.device)[labels]


def get_dataloaders(
    batch_size: int = 32, test_size: float = 0.2, noise: float = 0.2
):
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test, y_test


# --------------------
# 2. Model definition
# --------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.output_d = nn.Linear(16, 2) 
        self.output_R = nn.Linear(16, 1) 

    def forward(self, x: torch.Tensor):
        hidden_out = self.hidden(x)
        d = self.output_d(hidden_out) 
        R = self.output_R(hidden_out) 
        return d, R


# --------------------
# 3. Training loop
# --------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 1000,
    lr: float = 1e-3,
):
    model.to(device)
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            d_pred, R_pred = model(X)

            y_onehot = to_one_hot(y, num_classes=2)

            R = torch.norm(y_onehot, dim=1, keepdim=True) 
            d = y_onehot / (R + 1e-8) 

            loss1 = criterion_mse(R_pred * d_pred, y_onehot)

            loss2 = criterion_mse(R * d_pred, y_onehot)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {loss.item():.4f}, "
                f"Loss1: {loss1.item():.4f}, "
                f"Loss2: {loss2.item():.4f}"
            )


# --------------------
# 4. Plotting utilities
# --------------------
def plot_d_vectors_with_ambiguous(d_test: torch.Tensor, ambiguous_indices: np.ndarray):

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)

    d_test_np = d_test.cpu().numpy()

    plt.figure()
    plt.plot(unit_circle_x, unit_circle_y, "r--", label="Unit Circle")
    plt.scatter(d_test_np[:, 0], d_test_np[:, 1], alpha=0.7, label="d vectors")
    plt.scatter(
        d_test_np[ambiguous_indices, 0],
        d_test_np[ambiguous_indices, 1],
        color="r",
        edgecolor="k",
        label="Ambiguous d vectors",
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("Two_moons1.png", dpi=300, transparent=True)


def plot_decision_boundary_with_ambiguous(
    model: nn.Module,
    X_test: torch.Tensor,
    ambiguous_indices: np.ndarray,
):
    model.eval()
    X = X_test.cpu().numpy()

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = torch.meshgrid(
        torch.arange(x_min, x_max, 0.01),
        torch.arange(y_min, y_max, 0.01),
        indexing="ij",
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1).to(device)

    with torch.no_grad():
        d_grid, R_grid = model(grid)
        R_grid = R_grid.squeeze(1)
        Rd_grid = R_grid.unsqueeze(1) * d_grid
        Z = torch.argmax(torch.softmax(Rd_grid, dim=1), dim=1).reshape(xx.shape)

    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    Z_np = Z.cpu().numpy()

    plt.figure()
    plt.contourf(xx_np, yy_np, Z_np, alpha=0.5, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], color="b", alpha=0.7, label="All data")

    plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 0.1, 1))
    plt.yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 0.1, 1))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.scatter(
        X[ambiguous_indices, 0],
        X[ambiguous_indices, 1],
        color="r",
        edgecolor="k",
        label="Ambiguous points",
    )

    plt.legend()
    plt.savefig("Two_moons2.png", dpi=300, transparent=True)


def plot_uncertainty_vs_boundary_distance(
    model: nn.Module,
    X_test: torch.Tensor,
    threshold: float = 0.15,
    save_path: str = "Two_moons3.png",
):
    model.eval()
    X_test_cpu = X_test.cpu()

    with torch.no_grad():
        d_test, R_test = model(X_test_cpu.to(device))
    d_test = d_test.cpu()
    R_test = R_test.cpu()

    d_norms = torch.norm(R_test * d_test, dim=1)
    u_score = torch.abs(d_norms - 1).numpy()

    x_min, x_max = X_test_cpu[:, 0].min() - 0.5, X_test_cpu[:, 0].max() + 0.5
    y_min, y_max = X_test_cpu[:, 1].min() - 0.5, X_test_cpu[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        d_grid, R_grid = model(grid)
        R_grid = R_grid.squeeze(1)
        Rd_grid = R_grid.unsqueeze(1) * d_grid
        Z = torch.argmax(torch.softmax(Rd_grid, dim=1), dim=1).cpu().numpy()

    Z = Z.reshape(xx.shape)

    boundary_mask = np.zeros_like(Z, dtype=bool)
    boundary_mask[:-1, :] |= Z[:-1, :] != Z[1:, :]
    boundary_mask[:, :-1] |= Z[:, :-1] != Z[:, 1:]

    boundary_points = np.c_[xx[boundary_mask], yy[boundary_mask]]

    tree = cKDTree(boundary_points)
    dist_to_boundary, _ = tree.query(X_test_cpu.numpy(), k=1)

    ambiguous_idx = u_score > threshold
    non_ambiguous_idx = ~ambiguous_idx

    plt.figure(figsize=(6, 5))
    plt.scatter(
        u_score[non_ambiguous_idx],
        dist_to_boundary[non_ambiguous_idx],
        alpha=0.6,
        color="blue",
        label=f"u(x) ≤ {threshold}",
    )
    plt.scatter(
        u_score[ambiguous_idx],
        dist_to_boundary[ambiguous_idx],
        alpha=0.8,
        color="red",
        edgecolor="k",
        label=f"u(x) > {threshold}",
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()


# --------------------
# 5. Main
# --------------------
def main():
    set_seed(42)
    train_loader, test_loader, X_test, y_test = get_dataloaders()

    model = SimpleNN()
    train_model(model, train_loader, num_epochs=1000, lr=1e-3)

    X_test = X_test.to(device)
    with torch.no_grad():
        d_test, R_test = model(X_test)
    d_test_cpu = d_test.cpu()
    R_test_cpu = R_test.cpu()

    d_norms = torch.norm(R_test_cpu * d_test_cpu, dim=1)
    ambiguous_indices = (d_norms < 0.85).numpy()

    plot_d_vectors_with_ambiguous(d_test_cpu, ambiguous_indices)
    plot_decision_boundary_with_ambiguous(model, X_test.cpu(), ambiguous_indices)
    plot_uncertainty_vs_boundary_distance(model, X_test.cpu(), threshold=0.15)


if __name__ == "__main__":
    main()
