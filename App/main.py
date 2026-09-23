import argparse
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    #Return the best available device (CUDA if available, else CPU).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cwd = Path(__file__).resolve().parents[1]

def load_model(device: torch.device):

    model = torchvision.models.mobilenet_v3_small(pretrained=False)

    state_dict = torch.load(_cwd/'Model'/'weigths', map_location=device)

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    class_names = None

    return model, class_names


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image, apply the MobileNet preprocessing pipeline, and return a tensor.

    The function raises ``ValueError`` if the image cannot be opened or is not a valid RGB image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image '{image_path}': {e}")

    # MobileNetV2 expects 224x224 images; use the standard ImageNet transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # add batch dimension
    return img_tensor


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


def main():
    parser = argparse.ArgumentParser(description="Run MobileNet inference on an image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--output",
        default="Results",
        help="Directory to write the result file (default: Results)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image file '{image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    model, class_names = load_model(device)
    try:
        input_tensor = preprocess_image(str(image_path), device)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    probs = predict(model, input_tensor)
    score = generate_score(probs)
    pred_idx = probs.argmax().item()
    pred_name = class_names[pred_idx] if class_names else f"class_{pred_idx}"

    print(f"Predicted class: {pred_name} (index {pred_idx})")
    print(f"Confidence score: {score}%")

    # Write a simple result file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{image_path.stem}_result.txt"
    with result_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Predicted class: {pred_name}\n")
        f.write(f"Class index: {pred_idx}\n")
        f.write(f"Confidence score: {score}%\n")

    print(f"Result written to {result_path}")


if __name__ == "__main__":
    main()
