# IKT213-Project — Deepfake Image Detection

A deepfake detection system that classifies face images as **real** or **fake** using a fine-tuned **MobileNetV3-Small** model. Face detection is performed with OpenCV's **YuNet** DNN detector, and the application is available both as a CLI tool and a Flet desktop GUI.

## Project structure

```
IKT213-project/
├── App/
│   ├── main.py                  # CLI entry point (argparse)
│   ├── window.py                # Flet GUI entry point
│   ├── include/
│   │   ├── torch_utils.py       # Model loading, preprocessing, inference helpers
│   │   └── face_recognition.py  # YuNet face detection wrapper
├── Model/
│   ├── trainModel.py            # Training / evaluation script
│   ├── customDataset.py         # PyTorch Dataset for the image CSVs
│   ├── preprocess_dataset.py    # Dataset preprocessing (face-crop extraction)
│   ├── trueModel.pht.tar        # Trained model checkpoint
│   └── training_checkpoint.pht.tar  # Periodic training checkpoints
├── Results/                     # CLI inference output (one .txt per image)
└── pyproject.toml               # Project metadata and dependencies
```

## Pipeline overview

```mermaid
flowchart LR
    A[Input image] --> B[YuNet face detection<br/>OpenCV DNN]
    B --> C[Face crops RGB]
    C --> D[Preprocess<br/>Resize 256 → CenterCrop 224 → Normalize]
    D --> E[MobileNetV3-Small<br/>binary classifier]
    E --> F[real / fake + confidence %]
```

## The app

### CLI (`App/main.py`)

Runs inference on a single image, prints per-face results, and writes a result file.

```bash
# Basic usage
python App/main.py --image "Images/Fake faces/fake_0.png"
```

```bash
# Output to custom directory
python App/main.py --image "path/to/image.png" --output Results
```

Output (console):

```
Using device: cpu
Face 1: Predicted class: fake (confidence 100.0%)
Result written to Results/fake_0_result.txt
```

Result file format:

```
Image: Images/Fake faces/fake_0.png
Face 1: class=fake, index=1, confidence=100.0%
```

### GUI (`App/window.py`)

Flet desktop app with:

- Gradient background and a centered **Upload Image** button.
- Displays the original image with a classification label overlay.
- Shows a side panel of the detected face thumbnails.

```bash
python App/window.py
# or, via the Flet CLI:
flet run App/window.py
```

## Key modules

### `App/include/face_recognition.py`

- `detect_faces(img, model_yunet, score=0.6, nms_threshold=0.3, scale=1.0, index_backend_target_pair=0)` — runs YuNet (OpenCV `FaceDetectorYN`) and returns a list of cropped face images (BGR arrays).
- `get_detector(...)` — cached detector factory (`lru_cache`), parameterized by backend/target pair.
- `check_base_model()` — downloads `face_detection_yunet_2023mar.onnx` from Hugging Face if missing.

### `App/include/torch_utils.py`

- `get_device()` — CUDA if available, else CPU.
- `load_model(device)` — loads MobileNetV3-Small, replaces the final classifier with a 2-class head, loads `Model/trueModel.pht.tar`, returns `(model, class_names)` with `class_names = ["real", "fake"]`.
- `load_and_detect_faces(image_path)` — reads an image, detects faces with YuNet, returns RGB crops.
- `preprocess_faces_to_tensor(faces, device)` — batches the crops through MobileNetV3 transforms.
- `predict(model, input_tensor)` — evaluative forward pass returning softmax probabilities.
- `generate_score(prob)` — converts max probability to a 0–100 score.
- `predict_image(model, input_tensor)` — aggregate whole-image prediction based on max fake-probability across all faces.

## Development

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### Build (Flet)

```bash
flet build windows -v
```

See the [Flet docs](https://flet.dev/docs/) for other platforms (web, macOS, Linux, Android, iOS).

## Notes

- `torch` and `torchvision` are **not usable on the web** target (see `pyproject.toml`).
- When packaging with PyInstaller, the model checkpoint and YuNet ONNX file must be bundled (e.g. with `--add-binary`) and resolved at runtime from `sys._MEIPASS`.
