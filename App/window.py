from dataclasses import dataclass, field
from pathlib import Path

import flet as ft

# Import core functions from the CLI implementation
from include.torch_utils import (
    generate_score,
    get_device,
    load_model,
    predict,
    preprocess_image,
)


@dataclass
class State:
    picked_file_path: str | Path | None = field(default_factory=str)


_state = State()

def main(page: ft.Page):
    """Flet UI for the MobileNetV3 inference pipeline.

    The page shows a gradient background, an upload button in the centre,
    and displays the uploaded image together with a prediction rectangle and
    the confidence score.
    """
    # Page configuration
    page.title = "MobileNetV3 Inference"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window.width = 1280
    page.window.height = 720
    page.window.resizable = True

    # Load model once at startup
    device = get_device()
    model, class_names = load_model(device)

    # ---------------------------------------------------------------------
    # UI elements
    # ---------------------------------------------------------------------
    # Gradient background container – fills the whole page
    bg = ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.alignment.Alignment.TOP_LEFT,
            end=ft.alignment.Alignment.BOTTOM_RIGHT,
            colors=[ft.Colors.BLUE, ft.Colors.PURPLE],
        ),
    )

    # Placeholder for the image + overlay
    image_stack = ft.Stack(width=400, height=400)

    # Result text below the image
    result_text = ft.Text(value="", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)

    # ---------------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------------
    def on_file_picked(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        _state.picked_file_path = e.files[0].path
        run_inference(str(_state.picked_file_path))


    # File picker (hidden, used only for its dialog)
    file_picker = ft.FilePicker(
        on_result=on_file_picked
    )

    # Upload button – opens a file picker for images
    upload_btn = ft.Button(
        "Select image",
        action=ft.PickFiles(file_picker, allow_multiple=False),
        icon=ft.Icons.FOLDER_OPEN,
    )
    page.services.append(file_picker)

    # ---------------------------------------------------------------------
    # Helper: run inference and update UI
    # ---------------------------------------------------------------------
    def run_inference(image_path: str):
        try:
            # Preprocess and predict
            input_tensor = preprocess_image(image_path, device)
            probs = predict(model, input_tensor)
            score = generate_score(probs)
            pred_idx = probs.argmax().item()
            pred_name = class_names[pred_idx] if class_names else f"class_{pred_idx}"
        except Exception as exc:
            result_text.value = f"Error: {exc}"
            page.update()
            return

        # Build the overlay components
        img = ft.Image(
            src=image_path,
            width=400,
            height=400,
            fit=ft.Image.fit,
        )
        # Red rectangle around the image (prediction area)
        rect = ft.Container(
            width=400,
            height=400,
            border=ft.Border.all(width=4, color=ft.Colors.RED),
        )
        # Label with class name and confidence at the bottom‑left corner
        label = ft.Container(
            content=ft.Text(
                f"{pred_name} ({score}%)",
                color=ft.Colors.WHITE,
                size=18,
            ),
            alignment=ft.alignment.Alignment.BOTTOM_LEFT,
            padding=ft.Padding.all(5),
        )
        # Update the stack
        image_stack.controls = [img, rect, label]
        result_text.value = f"Predicted: {pred_name} – Confidence: {score}%"
        page.update()

    # ---------------------------------------------------------------------
    # Assemble the page – a Stack with background and the UI column
    # ---------------------------------------------------------------------
    page.add(
        ft.Stack(
            controls=[
                bg,
                ft.Column(
                    controls=[
                        upload_btn,
                        image_stack,
                        result_text,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
            ],
        )
    )

if __name__ == "__main__":
    ft.run(main)
