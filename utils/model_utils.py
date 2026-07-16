# ============================================================
# Model Profiling, Metrics, and Training/Evaluation Utilities
# ============================================================
# Contains everything needed to profile the model (parameter
# count, FLOPs, GPU memory, inference latency) and to run a
# full train / validate / test loop with early stopping.
# ============================================================

import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from tqdm import tqdm


# ============================================================
# Profiling
# ============================================================

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n📊 MODEL PARAMETERS")
    print(f"Total Parameters        : {total_params:,}")
    print(f"Trainable Parameters    : {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")

    return total_params, trainable_params, non_trainable_params


def get_model_flops(model, input_size=(16, 3, 256, 256), device='cuda'):
    """Requires `thop` (pip install thop)."""
    from thop import profile

    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2  # standard MACs -> FLOPs conversion

    print("\n⚙️ COMPUTATION")
    print(f"MACs  : {macs / 1e9:.3f} GMACs")
    print(f"FLOPs : {flops / 1e9:.3f} GFLOPs")

    return macs, flops


def get_memory_usage(model, input_size=(16, 3, 256, 256), device='cuda'):
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        _ = model(dummy_input)

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print("\n💾 GPU MEMORY")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")

    return peak_memory


def get_inference_time(model, input_size=(16, 3, 256, 256),
                        device='cuda', warmup=20, runs=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm-up (important for accurate GPU timing)
    for _ in range(warmup):
        _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    timings = []

    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            timings.append(end - start)

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1 / mean_time

    print("\n⏱️ INFERENCE PERFORMANCE")
    print(f"Mean Latency : {mean_time * 1000:.2f} ms")
    print(f"Std Dev      : {std_time * 1000:.2f} ms")
    print(f"FPS          : {fps:.2f}")

    return mean_time, std_time, fps


def full_model_profile(model, device='cuda'):
    model.to(device)

    print("\n🚀 STARTING FULL MODEL PROFILING\n")

    params = get_model_parameters(model)

    try:
        macs, flops = get_model_flops(model, device=device)
    except Exception:
        print("⚠️ Install thop (`pip install thop`) for FLOPs computation")
        macs, flops = None, None

    memory = get_memory_usage(model, device=device) if device == 'cuda' else None

    latency = get_inference_time(model, device=device)

    print("\n✅ PROFILING COMPLETE")

    return {
        "params": params,
        "macs_flops": (macs, flops),
        "memory_MB": memory,
        "latency": latency
    }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1


# ============================================================
# Train / Validate / Test loops
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    return epoch_loss, acc, prec, rec, f1


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    return epoch_loss, acc, prec, rec, f1


def test_model(model, loader, criterion, device, class_names=None, show_plot=True):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if show_plot:
        plt.show()

    return avg_loss, acc, prec, rec, f1, all_labels, all_preds


def run_training(model, train_loader, val_loader, test_loader, optimizer, criterion,
                  device, class_names=None, num_epochs=200, patience=10,
                  checkpoint_path='best_model.pth', plot_curves=True):
    """Full training loop with early stopping, checkpointing on best
    validation loss, and a final test-set evaluation."""

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Prec: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Prec: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"⏳ No improvement in val_loss for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    print("📥 Loaded the best saved model for final testing.")

    print("\n🔍 Evaluating on Test Set:")
    test_loss, test_acc, test_prec, test_rec, test_f1, all_labels, all_preds = test_model(
        model, test_loader, criterion, device, class_names)
    print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
          f"Prec: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

    corrects = torch.eq(torch.tensor(all_labels), torch.tensor(all_preds))
    num_correct = corrects.sum().item()
    total = len(corrects)
    print(f"✅ Correct Predictions: {num_correct}/{total} ({100 * num_correct / total:.2f}%)")

    if plot_curves:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(train_accuracies, label='Train Acc')
        plt.plot(val_accuracies, label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return train_losses, val_losses, train_accuracies, val_accuracies
