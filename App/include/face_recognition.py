import sys
from pathlib import Path

import cv2
import numpy as np

_model_yunet = 'face_detection_yunet_2023mar.onnx'
_cwd = Path(__file__).parent

def check_base_model():
    """Check if the base ONNX model file exists."""
    model_path = _cwd / _model_yunet
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        print(f"Downloading {_model_yunet}...")
        import urllib.request
        url = f"https://huggingface.co/opencv/face_detection_yunet/resolve/main/{_model_yunet}"
        result = urllib.request.urlretrieve(url, model_path)
        if result[1] is not None:
            raise RuntimeError(f"Failed to download model from {url}")
        print("Download completed.")
    return model_path

def detect_faces(img: np.ndarray, model_yunet = _model_yunet, score=0.6, nms_threshold=0.7, scale: float = 1.0):
    """
    Detects faces in an image using OpenCV's YuNet DNN face detector.
    Args:
        img: Input image in BGR format (OpenCV default).
        model_yunet: Path to the YuNet ONNX model file.
        score: Minimum confidence threshold for a detection to be kept.
        nms_threshold: Non-maximum suppression threshold (overlap allowed).
        scale: Downscale factor for the analysed image (0-1). Useful for
               speeding up detection on large frames.
    Returns:
        list[np.ndarray]: A list of cropped face images (as BGR arrays).
                           Empty list if no faces found.
    """

    if model_yunet == _model_yunet:
        model_yunet = str(check_base_model())

    height, width = img.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        model_yunet, '', (320, 320), score, nms_threshold
    )

    # Resize image for with scale factor
    small_width = round(width * scale)
    small_height = round(height * scale)
    if min(small_width, small_height) < 32:
        raise RuntimeError('Scaled image must be at least 32 by 32 pixels.')

    img_resized = cv2.resize(img, (small_width, small_height))
    detector.setInputSize((small_width, small_height))

    _, faces = detector.detect(img_resized)

    face_image_list = []

    if faces is not None:
        for (x, y, w, h) in faces:
            # Scale coordinates back to original image size
            x, y, w, h = round(x / scale), round(y / scale), round(w / scale), round(h / scale)
            # Extract face region
            face = img[int(y):int(y+h), int(x):int(x+w)]
            face_image_list.append(face)
    return face_image_list
