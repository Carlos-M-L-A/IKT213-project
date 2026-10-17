import argparse
import sys
from pathlib import Path

from include.torch_utils import (
    generate_score,
    get_device,
    load_model,
    predict,
    preprocess_image,
)


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
