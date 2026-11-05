# Results

This directory holds **output files produced by the CLI inference pipeline** (`App/main.py`).

## Contents

For every image processed with:

```bash
python App/main.py --image <path-to-image>
```

a text file is written here (or to the `--output` directory, default `Results`):

```
<image_stem>_result.txt
```

Example: `Images/Fake faces/fake_0.png` → `Results/fake_0_result.txt`

## Result file format

```
Image: <full input path>
Face 1: class=<real|fake>, index=<0|1>, confidence=<0.00–100.00>%
Face 2: class=<real|fake>, index=<0|1>, confidence=<0.00–100.00>%
...
```

- One `Face N:` line per detected face (in detection order).
- `class` is the human-readable label (`real` or `fake`).
- `index` is the class index (0 = real, 1 = fake).
- `confidence` is the softmax probability of the predicted class (0–100, 2 decimals).

## Notes

- The directory is created automatically if it does not exist.
- If no faces are detected in an image, the run fails with an error (see `load_and_detect_faces`) and no result file is produced.
- The GUI (`App/window.py`) does **not** write result files — it displays results in-window.
