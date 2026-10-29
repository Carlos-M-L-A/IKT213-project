import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import csv

import cv2
import pandas as pd

from App.include import face_recognition


ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "Images"
OUTPUT = IMAGES / "processed_faces"

# Adjust source CSV names if necessary
DATASETS = {
    "train": "training.csv",
    "validation": "validate.csv",
    "test": "test.csv",
}


def preprocess_dataset(split, csv_name):
    print(f"\nProcessing {split} dataset")

    annotations = pd.read_csv(IMAGES / csv_name)

    output_dir = OUTPUT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    failed = []

    for index, row in annotations.iterrows():
        relative_path = str(row.iloc[0])
        label = int(row.iloc[1])

        image_path = IMAGES / relative_path
        output_path = output_dir / f"face_{index:07d}.jpg"

        try:
            if not output_path.is_file():
                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError("Could not read image")

                faces = face_recognition.detect_faces(image)

                if not faces:
                    raise ValueError("No faces detected")

                # Use first detected face
                face = faces[0]

                if face.size == 0:
                    raise ValueError("Empty face crop")

                if not cv2.imwrite(str(output_path), face):
                    raise RuntimeError("Could not save face crop")

            processed.append([
                output_path.relative_to(IMAGES).as_posix(),
                label
            ])

        except (ValueError, RuntimeError, cv2.error) as e:
            failed.append([relative_path, str(e)])

        if (index + 1) % 1000 == 0:
            print(
                f"{split}: {index + 1}/{len(annotations)} "
                f"| Saved: {len(processed)} "
                f"| Failed: {len(failed)}",
                flush=True
            )

    # Write the new CSV after processing
    with open(
        IMAGES / f"{split}_faces.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label"])
        writer.writerows(processed)

    # Store list of failed images
    with open(
        OUTPUT / f"{split}_failed.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "error"])
        writer.writerows(failed)

    print(
        f"{split} complete! "
        f"Saved: {len(processed)}, Failed: {len(failed)}"
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split, csv_name in DATASETS.items():
        preprocess_dataset(split, csv_name)

    print("\nAll datasets processed!")


if __name__ == "__main__":
    main()
