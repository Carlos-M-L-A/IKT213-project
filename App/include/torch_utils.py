from pathlib import Path

import cv2
from . import face_recognition
import numpy as np
import torch
import torchvision
from torchvision import transforms


def get_device() -> torch.device:
    #Return the best available device (CUDA if available, else CPU).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cwd = Path(__file__).resolve().parents[2]

def load_model(device: torch.device):

    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=2)
    model.to(device)

    state_dict = torch.load(_cwd/'Model'/'trueModel.pht.tar', map_location=device, weights_only=True)

    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    # Number of output classes (the size of the final linear layer)
    num_classes = model.classifier[-1].out_features if hasattr(model.classifier, '__len__') else 0

    # Define class names: index 0 -> real, index 1 -> fake (binary classification)
    if num_classes == 2:
        class_names = ["real", "fake"]
    else:
        # Fallback to generic placeholder names
        class_names = [f"class_{i}" for i in range(num_classes)]

    return model, class_names

def load_and_detect_faces(image_path: str) -> list[np.ndarray]:
    """Load an image and return a list of detected face crops (RGB)."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to open image '{image_path}'")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_recognition.detect_faces(image_rgb)
    if not faces:
        raise ValueError(f"No faces detected in image '{image_path}'")
    return faces


def preprocess_faces_to_tensor(faces: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert face crops into a batched MobileNetV3-ready tensor."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
    ])

    tensors = [preprocess(face) for face in faces]
    return torch.stack(tensors).to(device)

def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image, detect faces and apply the MobileNet preprocessing pipeline, and return a tensor.

    The function raises ``ValueError`` if the image cannot be opened or is not a valid RGB image.
    """
    faces = load_and_detect_faces(image_path)
    return preprocess_faces_to_tensor(faces, device)


def predict(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    #Run a forward pass and return the softmax probabilities.
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.squeeze(0)  # shape: (num_classes,)


def generate_score(prob: torch.Tensor) -> float:
    #Convert the highest probability into a 0‑100 score.
    max_conf = prob.max().item()
    return round(max_conf * 100, 2)
