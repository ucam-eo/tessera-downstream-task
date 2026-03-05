#!/usr/bin/env python

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PatchRegressionDataset(Dataset):
    def __init__(self, patch_rep_files, patch_label_files):
        self.patch_rep_files = []
        self.patch_label_files = []

        logging.info("Scanning %d patch pairs...", len(patch_rep_files))
        for rep_file, label_file in tqdm(
            zip(patch_rep_files, patch_label_files), total=len(patch_rep_files), leave=False
        ):
            label_patch = np.load(label_file)
            if np.isnan(label_patch).all():
                continue
            self.patch_rep_files.append(rep_file)
            self.patch_label_files.append(label_file)

        logging.info("Kept %d valid patch pairs.", len(self.patch_rep_files))

    def __len__(self):
        return len(self.patch_rep_files)

    def __getitem__(self, idx):
        rep_patch = np.load(self.patch_rep_files[idx])
        label_patch = np.load(self.patch_label_files[idx])

        rep_patch = np.nan_to_num(rep_patch, nan=0.0)
        valid_mask = ~np.isnan(label_patch)
        label_patch = np.nan_to_num(label_patch, nan=0.0)

        rep_patch_tensor = torch.tensor(rep_patch, dtype=torch.float32).permute(2, 0, 1)
        label_patch_tensor = torch.tensor(label_patch, dtype=torch.float32)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)
        return rep_patch_tensor, label_patch_tensor, valid_mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetRegression(nn.Module):
    def __init__(self, in_channels=128, features=None, dropout=0.1):
        super().__init__()
        if features is None:
            features = [128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, dropout))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout)
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = self.ups[idx + 1](torch.cat((skip_connection, x), dim=1))

        return self.final_conv(x).squeeze(1)


def compute_regression_metrics(pred, target, valid_mask):
    valid_pred = pred[valid_mask].cpu().numpy()
    valid_target = target[valid_mask].cpu().numpy()
    if len(valid_pred) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mse = mean_squared_error(valid_target, valid_pred)
    mae = mean_absolute_error(valid_target, valid_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(valid_target, valid_pred)
    return mse, mae, rmse, r2


def evaluate(model, data_loader, criterion, device, split_name):
    model.eval()
    total_loss = 0.0
    batches = 0
    all_mse, all_mae, all_rmse, all_r2 = [], [], [], []

    with torch.no_grad():
        for rep, target, valid_mask in tqdm(data_loader, desc=f"{split_name}", leave=False):
            rep = rep.to(device)
            target = target.to(device)
            valid_mask = valid_mask.to(device)

            output = model(rep)
            loss_per_pixel = criterion(output, target)
            masked_loss = loss_per_pixel * valid_mask.float()
            valid_pixel_count = valid_mask.sum()

            if valid_pixel_count > 0:
                loss = masked_loss.sum() / valid_pixel_count
                total_loss += loss.item()
                batches += 1

                mse, mae, rmse, r2 = compute_regression_metrics(output, target, valid_mask)
                all_mse.append(mse)
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_r2.append(r2)

    if batches == 0:
        return float("inf"), float("inf"), float("inf"), float("inf"), -float("inf")

    return (
        total_loss / batches,
        float(np.mean(all_mse)),
        float(np.mean(all_mae)),
        float(np.mean(all_rmse)),
        float(np.mean(all_r2)),
    )


def train_model(model, train_loader, val_loader, device, epochs, lr, checkpoint_path):
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    best_val_r2 = -float("inf")
    best_epoch = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for rep, target, valid_mask in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} Train", leave=False
        ):
            rep = rep.to(device)
            target = target.to(device)
            valid_mask = valid_mask.to(device)

            optimizer.zero_grad()
            output = model(rep)
            loss_per_pixel = criterion(output, target)
            masked_loss = loss_per_pixel * valid_mask.float()
            valid_pixel_count = valid_mask.sum()
            loss = (
                masked_loss.sum() / valid_pixel_count
                if valid_pixel_count > 0
                else torch.tensor(0.0, requires_grad=True, device=device)
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        val_loss, val_mse, val_mae, val_rmse, val_r2 = evaluate(
            model, val_loader, criterion, device, "Validation"
        )
        scheduler.step(val_loss)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                },
                checkpoint_path,
            )

        logging.info(
            "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | Val R2: %.4f | Val MSE: %.4f | Val MAE: %.4f | Val RMSE: %.4f",
            epoch + 1,
            epochs,
            avg_train_loss,
            val_loss,
            val_r2,
            val_mse,
            val_mae,
            val_rmse,
        )

        if epoch - best_epoch > 20:
            logging.info("Early stopping at epoch %d", epoch + 1)
            break


def collect_region_files(base_data_dir, region_id):
    region_dir = base_data_dir / f"region_{region_id}"
    if not region_dir.exists():
        return [], []

    rep_files = sorted(region_dir.glob("patch_*.npy"))
    label_files = sorted(region_dir.glob("label_*.npy"))
    return [str(p) for p in rep_files], [str(p) for p in label_files]


def get_split_files(base_data_dir, train_regions, val_region, test_region):
    train_rep_files, train_label_files = [], []
    for region_id in train_regions:
        rep_files, label_files = collect_region_files(base_data_dir, region_id)
        train_rep_files.extend(rep_files)
        train_label_files.extend(label_files)

    val_rep_files, val_label_files = collect_region_files(base_data_dir, val_region)
    test_rep_files, test_label_files = collect_region_files(base_data_dir, test_region)
    return (
        train_rep_files,
        train_label_files,
        val_rep_files,
        val_label_files,
        test_rep_files,
        test_label_files,
    )


def main():
    parser = argparse.ArgumentParser(description="Train UNet for Borneo canopy-height regression")
    parser.add_argument("--train_regions", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val_region", type=int, default=3)
    parser.add_argument("--test_region", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/borneo/training_data_tessera_embedding"), # Tessera embedding
        # default=Path("data/borneo/training_data_presto_embedding"), # Presto embedding
        # default=Path("data/borneo/training_data_efm_embedding"), # Google Satellite embedding (AlphaEarth)
        help="Path relative to project root unless an absolute path is given.",
    )
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    project_root = Path(__file__).resolve().parents[1]
    base_data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    if not base_data_dir.exists():
        raise ValueError(f"Data directory does not exist: {base_data_dir}")

    (
        train_rep_files,
        train_label_files,
        val_rep_files,
        val_label_files,
        test_rep_files,
        test_label_files,
    ) = get_split_files(base_data_dir, args.train_regions, args.val_region, args.test_region)

    if not train_rep_files:
        raise ValueError("No training representation files found.")

    if args.train_ratio < 1.0:
        sample_count = int(len(train_rep_files) * args.train_ratio)
        selected = np.random.choice(len(train_rep_files), sample_count, replace=False)
        selected = sorted(selected)
        train_rep_files = [train_rep_files[i] for i in selected]
        train_label_files = [train_label_files[i] for i in selected]

    sample_rep_patch = np.nan_to_num(np.load(train_rep_files[0]), nan=0.0)
    _, _, patch_c = sample_rep_patch.shape

    train_dataset = PatchRegressionDataset(train_rep_files, train_label_files)
    val_dataset = PatchRegressionDataset(val_rep_files, val_label_files)
    test_dataset = PatchRegressionDataset(test_rep_files, test_label_files)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering invalid labels.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRegression(in_channels=patch_c, features=[128, 256, 512], dropout=0.1).to(device)
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = project_root / "checkpoints" / "borneo_regression" / run_tag / "best_model.pth"

    train_model(model, train_loader, val_loader, device, args.epochs, args.lr, checkpoint_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.MSELoss(reduction="none")
    test_loss, test_mse, test_mae, test_rmse, test_r2 = evaluate(
        model, test_loader, criterion, device, "Test"
    )

    logging.info("Final test metrics:")
    logging.info("Loss: %.6f | MSE: %.4f | MAE: %.4f | RMSE: %.4f | R2: %.4f", test_loss, test_mse, test_mae, test_rmse, test_r2)
    print("\nFinal test metrics:")
    print(f"Loss: {test_loss:.6f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
