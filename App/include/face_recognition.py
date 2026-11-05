import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=8)
def get_detector(
    model_path: str,
    score: float = 0.6,
    nms_threshold: float = 0.3,
    index_backend_target_pair: int = 0
):
    backend_id, target_id = BACKEND_TARGET_PAIRS[
        index_backend_target_pair
    ]

    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=score,
        nms_threshold=nms_threshold,
        backend_id=backend_id,
        target_id=target_id
    )

    return detector


_model_yunet = 'face_detection_yunet_2023mar.onnx'
_cwd = Path(__file__).parent

# Backend and target device pairs for OpenCV DNN
BACKEND_TARGET_PAIRS = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

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

def detect_faces(img: np.ndarray, model_yunet = _model_yunet, score=0.6, nms_threshold=0.3, scale: float = 1.0, index_backend_target_pair: int = 0):
    """
    Detects faces in an image using OpenCV's YuNet DNN face detector.
    Args:
        img: Input image in BGR format (OpenCV default).
        model_yunet: Path to the YuNet ONNX model file.
        score: Minimum confidence threshold for a detection to be kept.
        nms_threshold: Non-maximum suppression threshold (overlap allowed).
        scale: Downscale factor for the analysed image (0-1). Useful for
               speeding up detection on large frames.
        index_backend_target_pair: Index of the backend/target pair to use.
    Returns:
        list[np.ndarray]: A list of cropped face images (as BGR arrays).
                           Empty list if no faces found.
    """

    if model_yunet == _model_yunet:
        model_yunet = str(check_base_model())

    height, width = img.shape[:2]

    if scale <= 0 or scale > 1:
        raise ValueError("scale must be between 0 and 1")

    # Resize image for with scale factor
    small_width = round(width * scale)
    small_height = round(height * scale)

    if min(small_width, small_height) < 32:
        raise RuntimeError('Scaled image must be at least 32 by 32 pixels.')

    # Reuse existing detector
    detector = get_detector(
        str(model_yunet),
        score,
        nms_threshold,
        index_backend_target_pair
    )


    img_resized = cv2.resize(img, (small_width, small_height))
    detector.setInputSize((small_width, small_height))

    _, faces = detector.detect(img_resized)

    face_image_list = []

    if faces is None:
        return face_image_list

    # Convert coordinates
    for face in faces:

        x, y, w, h = face[:4] / scale

        #bounding box corners
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w))
        y2 = int(np.ceil(y + h))

        #Clamp coordinates to image boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        #Skip invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        #Crop face
        face_image = img[y1:y2, x1:x2]

        #check crop is valid
        if face_image.size == 0:
            continue

        face_image_list.append(face_image)

    return face_image_list
