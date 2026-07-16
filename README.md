# GILF: Gated Iterative Learnable Lifting Fusion for WBC Classification

A dual-backbone (MobileNetV2 + DenseNet201) image classification model that
fuses features using a novel **Gated Iterative Learnable Lifting Fusion
(GILF)** module, with Grad-CAM++ interpretability. Originally built for
white blood cell (WBC) classification on the PBC dataset (8 classes), but
the architecture is dataset-agnostic — point it at any `ImageFolder`-style
directory.

## How the pieces fit together

The model is a pipeline of five stages. Each stage lives in its own file so
you can read, test, or swap out any piece independently.

```
Input image (3, H, W)
        │
        ├──► MobileNetV2 + SE  ──► feature map A_raw  (1280 ch)
        │        (backbones.py)
        │
        └──► DenseNet201 + CBAM ──► feature map B_raw (1920 ch)
                 (backbones.py)

A_raw, B_raw
        │
        ▼
Channel Alignment (channel_alignment.py)
   A_raw → 512 ch,  B_raw → 512 ch
        │
        ▼
Gated Iterative Lifting Fusion (gilf_fusion.py)
   T iterative refinement steps, bidirectional gated updates
        │
        ▼
fused feature map (512 ch)
        │
        ▼
Global Average Pool → Dropout → Linear  (assembled in fusion_model.py)
        │
        ▼
     logits (num_classes)
```

`fusion_model.py` is the file that wires stages 1–5 together into a single
`nn.Module` (`LiftingFusionNet`), so it imports from `backbones.py`,
`channel_alignment.py`, and `gilf_fusion.py`.

`data_loader.py`, `utils/model_utils.py`, and `gradcam.py` are used around
the model — for getting data in, measuring/training it, and explaining its
predictions, respectively.

## Repository structure

```
gilf-wbc-classification/
├── README.md
├── requirements.txt
├── train.py                     # main entry point — run this
├── data_loader.py                # BLOCK: data loading & stratified split
├── models/
│   ├── __init__.py
│   ├── backbones.py              # BLOCK 1 + 2: MobileNetV2+SE, DenseNet201+CBAM
│   ├── channel_alignment.py       # BLOCK 3: project both branches to 512-d
│   ├── gilf_fusion.py            # BLOCK 4: Gated Iterative Lifting Fusion
│   └── fusion_model.py           # BLOCK 5: full end-to-end LiftingFusionNet
├── utils/
│   ├── __init__.py
│   └── model_utils.py            # profiling (params/FLOPs/memory/latency)
│                                  # + metrics + train/validate/test loops
└── gradcam.py                    # Grad-CAM++ implementation & visualization
```

| File | What it contains | Depends on |
|---|---|---|
| `data_loader.py` | `get_dataloaders()` — stratified 70/20/10 split, augmentation, class-balanced sampling | torchvision, sklearn |
| `models/backbones.py` | `MobileNetV2SE_FeatureEncoder`, `DenseNet201_CBAM_FeatureEncoder`, plus their SE/CBAM sub-blocks | torchvision |
| `models/channel_alignment.py` | `ChannelAlign` — 1×1 conv + BN projection to a shared channel dim | torch |
| `models/gilf_fusion.py` | `GatedIterativeLiftingFusion` — the core fusion idea (shared predictor, direction adapters, gated updates, T iterations) | torch |
| `models/fusion_model.py` | `LiftingFusionNet` — assembles all of the above into one model | the three files above |
| `utils/model_utils.py` | `full_model_profile`, `compute_metrics`, `train_one_epoch`, `validate_one_epoch`, `test_model`, `run_training` | torch, sklearn, thop, tqdm |
| `gradcam.py` | `GradCAMPlusPlus`, `visualize_gradcam` | opencv-python |
| `train.py` | CLI script that runs the whole pipeline end to end | all of the above |

## Setup

```bash
git clone <your-repo-url>
cd gilf-wbc-classification
pip install -r requirements.txt
```

Your dataset should be laid out as a standard `torchvision.datasets.ImageFolder`
directory, i.e. one sub-folder per class:

```
PBC_dataset_normal_DIB/
├── class_1/
│   ├── img001.jpg
│   └── ...
├── class_2/
│   └── ...
└── ...
```

## Running everything

The simplest path is the provided CLI script:

```bash
python train.py --data_dir /path/to/PBC_dataset_normal_DIB \
                 --num_classes 8 \
                 --num_epochs 50 \
                 --batch_size 16
```

This will, in order:
1. Build stratified train/val/test loaders (`data_loader.py`)
2. Construct `LiftingFusionNet` (`models/fusion_model.py`)
3. Print a full profile: parameter count, FLOPs, GPU memory, latency (`utils/model_utils.py`)
4. Train with early stopping, saving the best checkpoint to `--checkpoint` (default `best_model.pth`)
5. Evaluate on the test set and plot loss/accuracy curves
6. Run Grad-CAM++ on a handful of test images and show the overlays (`gradcam.py`)

Useful flags: `--skip_profile`, `--skip_gradcam`, `--lr`, `--dropout`, `--T`
(number of lifting-fusion iterations), `--seed`.

### Using the pieces individually

Every module is also runnable and importable on its own, e.g.:

```python
from models.fusion_model import LiftingFusionNet
import torch

model = LiftingFusionNet(num_classes=8, dropout_p=0.3, T=3)
x = torch.randn(2, 3, 256, 256)
logits = model(x)
```

```python
from data_loader import get_dataloaders

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir="/path/to/dataset", batch_size=16
)
```

Each file under `models/` also has a small `if __name__ == "__main__":`
sanity check at the bottom — run e.g. `python -m models.backbones` to verify
shapes independently of the rest of the pipeline.

## Notes on the architecture

**Why two backbones?** MobileNetV2 (lightweight, efficient) and DenseNet201
(deep, dense feature reuse) capture complementary characteristics; SE and
CBAM attention sharpen each backbone's own features before fusion.

**Why "lifting"?** The fusion module borrows the update/predict structure
of the classical wavelet lifting scheme: each branch predicts the other,
the residual (prediction error) is used to update the original branch, and
this is repeated for `T` iterations with a learned spatial gate controlling
how much each branch updates the other at every step.

**Grad-CAM++ target layer.** By default, `visualize_gradcam()` hooks into
`model.fusion` (the output of the GILF module) so the heatmap reflects the
*fused* representation actually used for classification, not just one
backbone's features.

## Suggested additions before publishing

A few things worth adding depending on how you intend to share this:
- **A `LICENSE` file** (MIT/Apache-2.0 are common for research code).
- **Sample outputs** (loss/accuracy curves, a Grad-CAM++ grid, a confusion
  matrix) in a `results/` or `assets/` folder, referenced from this README.
- **A `config.yaml` or similar** if you plan to run many experiments with
  different hyperparameters, instead of long CLI commands.
- **A `.gitignore`** (included) so checkpoints, `__pycache__/`, and dataset
  folders aren't accidentally committed.
- **Citation info** if this accompanies a paper — add a `CITATION.cff` or a
  BibTeX snippet at the bottom of this README.
- Do **not** commit the dataset itself or your trained `.pth` checkpoint if
  either is large or not yours to redistribute; link to the dataset source
  instead (e.g. the Kaggle PBC dataset page).
