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
    # Determine if we have a batch of faces
    if probs.dim() == 1:
        # Single face – keep existing logic
        scores = [generate_score(probs)]
        pred_idxs = [probs.argmax().item()]
    else:
        # Multiple faces – compute per‑face scores and class indices
        scores = [generate_score(p) for p in probs]
        pred_idxs = [p.argmax().item() for p in probs]
    # Map indices to class names (fallback to generic names)
    pred_names = [class_names[idx] if class_names else f"class_{idx}" for idx in pred_idxs]

    # Print results for each detected face (or the single face)
    for i, (name, sc) in enumerate(zip(pred_names, scores)):
        print(f"Face {i+1}: Predicted class: {name} (confidence {sc}%)")

    # Write a simple result file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{image_path.stem}_result.txt"
    with result_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        # Write a line per face
        for i, (name, idx, sc) in enumerate(zip(pred_names, pred_idxs, scores)):
            f.write(f"Face {i+1}: class={name}, index={idx}, confidence={sc}%\n")

    print(f"Result written to {result_path}")


if __name__ == "__main__":
    main()
