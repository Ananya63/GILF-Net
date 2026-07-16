# ============================================================
# Main Entry Point
# ============================================================
# Ties together data loading, model construction, profiling,
# training, testing, and Grad-CAM++ visualization.
#
# Usage:
#   python train.py --data_dir /path/to/PBC_dataset_normal_DIB
# ============================================================

import argparse
import random

import numpy as np
import torch
import torch.optim as optim

from data_loader import get_dataloaders
from models.fusion_model import LiftingFusionNet
from utils.model_utils import full_model_profile, run_training
from gradcam import visualize_gradcam


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train the GILF fusion model")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the ImageFolder-style dataset root")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--T", type=int, default=3, help="Lifting fusion iterations")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--skip_gradcam", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        save_split_csv="classwise_split.csv"
    )

    # 2) Model
    model = LiftingFusionNet(
        num_classes=args.num_classes,
        dropout_p=args.dropout,
        T=args.T
    ).to(device)

    # 3) Profiling (optional)
    if not args.skip_profile:
        full_model_profile(model, device=device.type)

    # 4) Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        class_names=class_names,
        num_epochs=args.num_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint
    )

    # 5) Grad-CAM++ visualization (optional)
    if not args.skip_gradcam:
        visualize_gradcam(
            model=model,
            loader=test_loader,
            device=device,
            class_names=class_names,
            num_images=6
        )


if __name__ == "__main__":
    main()
