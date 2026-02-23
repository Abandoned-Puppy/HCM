#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



def generate_data(mode: str = "gaussian_hetero",
                  train: bool = True,
                  n_points: int = 4000):
    """
    mode:
        - 'gaussian_hetero' : y = x^3 + heteroscedastic Gaussian noise
        - 'laplace_hetero'  : Laplace heteroscedastic noise
        - 'mixture'         : bimodal Gaussian mixture (in |x|<=2)
        - 'multi_y_x2'      : y = ±sqrt(x) + noise (multi-valued)
    """
    if train:
        x = np.linspace(-4.0, 4.0, n_points)
    else:
        x = np.linspace(-6.0, 6.0, n_points)

    y_clean = x ** 3

    sigma = np.zeros_like(x)
    mask = (x >= -2.0) & (x <= 2.0)
    sigma_max = 5.0
    sigma[mask] = (x[mask] + 2.0) / 4.0 * sigma_max

    if train:
        if mode == "gaussian_hetero":
            eps = np.random.normal(loc=0.0, scale=sigma)

        elif mode == "laplace_hetero":
            b = sigma / np.sqrt(2.0)
            eps = np.random.laplace(loc=0.0, scale=b)

        elif mode == "mixture":
            eps = np.zeros_like(x)
            comp = np.random.choice([-1, 1], size=n_points)
            mu_scale = 3.0
            mu = mu_scale * sigma 
            eps[mask] = np.random.normal(
                loc=comp[mask] * mu[mask],
                scale=sigma[mask],
            )

        elif mode == "multi_y_x2":
            x = np.linspace(0.0, 4.0, n_points)
            y_plus = np.sqrt(x)
            y_minus = -np.sqrt(x)
            branch = np.random.choice([-1, 1], size=n_points)
            y_clean = np.where(branch > 0, y_plus, y_minus)
            sigma = 0.2 + 0.3 * (x / 4.0)
            eps = np.random.normal(0.0, sigma)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y = y_clean + eps

    else:
        if mode == "multi_y_x2":
            x = np.linspace(0.0, 4.0, n_points)
            y = np.zeros_like(x)
        else:
            y = y_clean

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return x_tensor, y_tensor



class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)   
        self.fc5 = nn.Linear(100, 1)  
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        d = self.fc4(x)      
        R = self.fc5(x)     
        return R, d



def train(model, train_loader, device, epochs=1000, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[200, 400, 600, 800],
        gamma=0.1,
    )

    model.to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred_R, pred_d = model(x)

            y_expanded = torch.cat([y, y], dim=1)  


            R_target = torch.sqrt(torch.sum(y_expanded ** 2, dim=1, keepdim=True))  
            d_target = y_expanded / (R_target + 1e-8)                               

            d_loss = criterion(pred_R * d_target, y_expanded)
            R_loss = criterion(R_target * pred_d, y_expanded)

            loss = d_loss + R_loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"R Loss: {R_loss.item():.4f} | d Loss: {d_loss.item():.4f}"
        )


@torch.no_grad()
def evaluate_and_plot(model,
                      test_loader,
                      x_test,
                      y_test,
                      device,
                      outfig="hcm_1d_result.png"):
    model.eval()
    model.to(device)

    conf_list = []
    pred_list = []

    for x, _ in test_loader:
        x = x.to(device)
        pred_R, pred_d = model(x)


        d_norm_sq = torch.sum(pred_d ** 2, dim=1)          
        conf = torch.sqrt(torch.abs(d_norm_sq - 1.0))      
        conf = conf * pred_R.squeeze(-1)                   
        pred_y = (pred_R * pred_d)[:, 0]                   

        conf_list.append(conf.cpu())
        pred_list.append(pred_y.cpu())

    conf_all = torch.cat(conf_list, dim=0)        
    pred_all = torch.cat(pred_list, dim=0)       

    x_np = x_test.squeeze(1).numpy()
    y_np = y_test.squeeze(1).numpy()
    pred_np = pred_all.numpy()
    conf_np = conf_all.numpy()


    plt.figure(figsize=(10, 6))
    plt.plot(x_np, y_np, label="Clean function f(x)")
    plt.scatter(
        x_test.squeeze(1).numpy(),
        y_test.squeeze(1).numpy(),
        label="Test data",
        s=5,
        alpha=0.4,
    )
    plt.plot(x_np, pred_np, label="Prediction", color="C1")

    for k in [1, 2, 3]:
        upper = pred_np + k * conf_np
        lower = pred_np - k * conf_np
        plt.fill_between(
            x_np,
            lower,
            upper,
            color="skyblue",
            alpha=0.15,
            label=f"{k}× uncertainty" if k == 1 else None,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outfig) or ".", exist_ok=True)
    plt.savefig(outfig, dpi=200)
    plt.close()
    print(f"[Saved] figure saved to: {outfig}")


# =========================
# 5. main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="1D regression toy example with hyperspherical decomposition."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixture",
        choices=["gaussian_hetero", "laplace_hetero", "mixture", "multi_y_x2"],
        help="data generation mode",
    )
    parser.add_argument(
        "--train_points",
        type=int,
        default=4000,
        help="number of training points",
    )
    parser.add_argument(
        "--test_points",
        type=int,
        default=2000,
        help="number of test points",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="training epochs",
    )
    parser.add_argument(
        "--outfig",
        type=str,
        default="hcm_1d_result.png",
        help="output figure path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) 데이터 생성
    x_train, y_train = generate_data(
        mode=args.mode,
        train=True,
        n_points=args.train_points,
    )
    x_test, y_test = generate_data(
        mode=args.mode,
        train=False,
        n_points=args.test_points,
    )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train.numpy())
    x_test_norm = scaler.transform(x_test.numpy())
    x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train)
    test_dataset = TensorDataset(x_test_tensor, y_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SimpleNN()
    train(model, train_loader, device, epochs=args.epochs, lr=1e-3)

    evaluate_and_plot(
        model,
        test_loader,
        x_test,
        y_test,
        device,
        outfig=args.outfig,
    )


if __name__ == "__main__":
    main()
