# Model — Training & Evaluation

This directory contains the training pipeline for the deepfake classifier and the resulting model artifacts.

## Files

| File | Purpose |
|------|---------|
| `trainModel.py` | Main training/evaluation script (finetunes MobileNetV3-Small). |
| `customDataset.py` | PyTorch `Dataset` that reads image paths + labels from CSV files. |
| `preprocess_dataset.py` | Extracts face crops from the raw dataset using YuNet, writes new CSVs. |
| `trueModel.pht.tar` | Final trained checkpoint (state_dict + optimizer + loss + accuracy). |
| `training_checkpoint.pht.tar` | Periodic checkpoint saved every 3 epochs during training. |
| `README.md` | This file. |

## Dataset conventions

- CSV rows: `image,label` where `label` is `0` for **real** and `1` for **fake**.
- Image paths are relative to `Images/`.
- Expected CSV names: `training.csv`, `validate.csv`, `test.csv` (used by `preprocess_dataset.py`), producing `train_faces.csv`, `validation_faces.csv`, `test_faces.csv` for training.

## Preprocessing (`preprocess_dataset.py`)

Before training, run the face-crop extraction:

```bash
python Model/preprocess_dataset.py
```

- Reads `Images/training.csv`, `Images/validate.csv`, `Images/test.csv`.
- For each image: detects faces with YuNet (`App/include/face_recognition.py`), saves the **first** detected face crop to `Images/processed_faces/<split>/face_0000000.jpg`.
- Writes `<split>_faces.csv` (processed images + original labels) and `<split>_failed.csv` (records that failed).
- Skips already-processed images (idempotent).

## Training (`trainModel.py`)

```bash
python Model/trainModel.py
```

Highlights:

- **Model:** `torchvision.models.mobilenet_v3_small(weights=IMAGENET1K_V1)`, classifier head replaced with `nn.Linear(..., out_features=2)`.
- **Data:** `ImageDataset` over `train_faces.csv`, `validation_faces.csv`, `test_faces.csv`, batch size 64.
- **Training:** Adam (lr 0.001), CrossEntropyLoss, 40 epochs. Checkpoint saved every 3 epochs.
- **Evaluation:** prints accuracy on train and test sets, builds a confusion matrix, and reports `real`/`fake` per-class accuracy.
- **Output:** saves `Model/trueModel.pht.tar` containing:
  ```python
  {'state_dict': model.state_dict(),
   'optimizer': optimizer.state_dict(),
   'loss': last_epoch_loss,
   'accuracy': test_accuracy}
  ```

## Model card (summary)

| Property | Value |
|----------|-------|
| Architecture | MobileNetV3-Small (ImageNet-pretrained) |
| Classification head | 2 classes: `real` (0), `fake` (1) |
| Input size | 224 × 224 (RGB, ImageNet normalization) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam, lr = 0.001 |
| Epochs | 40 |
| Checkpoint | `trueModel.pht.tar` |

## Inference

The trained checkpoint is consumed by `App/include/torch_utils.py` (`load_model`), which loads the `state_dict` with `strict=True` and maps `class_names = ["real", "fake"]`.

> **Note:** the checkpoint contains keys beyond the state dict (`optimizer`, `loss`, `accuracy`); `load_model` reads `checkpoint['state_dict']` for the weights.
