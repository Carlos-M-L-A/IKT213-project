from dataclasses import dataclass, field
from pathlib import Path

import flet as ft

import cv2
import base64

try:
    from include.torch_utils import (
        generate_score,
        get_device,
        load_model,
        predict,
        preprocess_image,
        load_and_detect_faces,
        preprocess_faces_to_tensor
    )

    INFERENCE_IMPORT_ERROR = None
except RuntimeError as exc:
    generate_score = get_device = load_model = predict = preprocess_image = load_and_detect_faces = preprocess_faces_to_tensor = None
    INFERENCE_IMPORT_ERROR = exc

@dataclass
class State:
    picked_file_path: str | Path | None = field(default_factory=str)

_state = State()

def main(page: ft.Page):
    page.title = "MobileNet Image Classifier"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    model = None
    class_names = []
    device = None
    inference_error = INFERENCE_IMPORT_ERROR

    if inference_error is None:
        try:
            device = get_device()
            model, class_names = load_model(device)
        except RuntimeError as exc:
            inference_error = exc

    # The UI will show the original image on the left and a column of detected faces on the right.
    image_row = ft.Row(width=820, height=420, alignment=ft.MainAxisAlignment.CENTER, spacing=20)
    result_text = ft.Text(
        value="Choose an image to begin.",
        size=20,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )
    file_picker = ft.FilePicker()

    def show_image(source, faces_bytes=None, label_text=""):
        # Original image with optional red overlay and label
        original = ft.Stack(
            width=400,
            height=400,
            controls=[
                ft.Image(
                    src=source,
                    width=400,
                    height=400,
                    fit=ft.BoxFit.CONTAIN,
                    error_content=ft.Text("Could not display this image"),
                ),
                ft.Container(
                    width=400,
                    height=400,
                    content=ft.Text(label_text, color=ft.Colors.WHITE, size=18),
                    alignment=ft.Alignment.BOTTOM_LEFT,
                    padding=ft.Padding.all(8),
                ),
            ],
        )
        if label_text:
            overlay = ft.Container(
                width=400,
                height=400,
                content=ft.Text(label_text, color=ft.Colors.WHITE, size=18),
                alignment=ft.Alignment.BOTTOM_LEFT,
                padding=ft.Padding.all(8),
            )
            original.controls.append(overlay)

        # Build the column of face thumbnails if any are supplied
        face_column = ft.Column(width=400, height=400, scroll=ft.ScrollMode.AUTO, spacing=5)
        if faces_bytes:
            face_images = []
            for b64 in faces_bytes:
                face_images.append(
                    ft.Image(
                        src=b64,
                        width=120,
                        height=120,
                        fit=ft.BoxFit.COVER,
                    )
                )
            face_column.controls = face_images
        else:
            # placeholder when no faces are detected
            face_column.controls = [ft.Text("No faces detected", color=ft.Colors.WHITE, size=16, text_align=ft.TextAlign.CENTER)]

        # Update the row container that holds both columns
        image_row.controls = [original, face_column]

    async def choose_image():
        files = await file_picker.pick_files(
            dialog_title="Select an image",
            file_type=ft.FilePickerFileType.IMAGE,
            allow_multiple=False,
            with_data=True,
        )
        if not files:
            return

        selected = files[0]
        _state.picked_file_path = selected.path
        image_source = selected.bytes if selected.bytes is not None else selected.path
        if image_source is None:
            result_text.value = "The selected image could not be read. Try another file."
            page.update()
            return

        # Show the original image immediately (no overlay yet)
        show_image(image_source)

        # Run inference (if possible) and face detection in parallel
        if inference_error is not None:
            result_text.value = f"Image loaded. Inference is unavailable: {inference_error}"
            page.update()
            return

        # --- Classification -------------------------------------------------
        try:
            input_tensor = preprocess_image(str(selected.path), device)
            probs = predict(model, input_tensor)
            score = generate_score(probs)
            pred_idx = probs.argmax().item()
            pred_name = class_names[pred_idx] if class_names else f"class_{pred_idx}"
            classification_label = f"{pred_name} ({score}%)"
        except Exception as exc:
            classification_label = f"Inference error: {exc}"

        # --- Face detection ------------------------------------------------
        face_bytes = []
        try:
            # Load faces as numpy arrays (RGB) using the helper from torch_utils
            faces = load_and_detect_faces(str(selected.path))
            for face in faces:
                # Convert RGB numpy array to BGR for OpenCV encoding
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode('.png', face_bgr)
                if success:
                    b64 = base64.b64encode(buffer.tobytes()).decode()
                    face_bytes.append(f"data:image/png;base64,{b64}")
        except Exception as exc:
            # If face detection fails we just keep face_bytes empty and will show a placeholder.
            classification_label = f"{classification_label} – Face detection error: {exc}"

        # Update UI with original image, detection results and classification label
        show_image(image_source, faces_bytes=face_bytes, label_text=classification_label)
        result_text.value = f"Predicted: {classification_label}"
        page.update()

    upload_btn = ft.Button(
        content="Select image",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=choose_image,
    )

    page.add(
        ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=[ft.Colors.BLACK, ft.Colors.PURPLE],
            ),
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
                controls=[upload_btn, image_row, result_text],
            ),
        )
    )


ft.run(main)
