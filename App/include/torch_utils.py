from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    #Return the best available device (CUDA if available, else CPU).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_input_size = 1024
_cwd = Path(__file__).resolve().parents[2]

def load_model(device: torch.device):

    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(in_features=_input_size, out_features=2)
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


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image, apply the MobileNet preprocessing pipeline, and return a tensor.

    The function raises ``ValueError`` if the image cannot be opened or is not a valid RGB image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError as e:
        raise ValueError(f"Failed to open image '{image_path}': {e}")

    # MobileNetV3 expects 224x224 images; use the standard ImageNet transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
    return img_tensor.to(device)


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
