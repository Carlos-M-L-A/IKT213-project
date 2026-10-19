from dataclasses import dataclass, field
from pathlib import Path

import flet as ft

try:
    from include.torch_utils import (
        generate_score,
        get_device,
        load_model,
        predict,
        preprocess_image,
    )

    INFERENCE_IMPORT_ERROR = None
except Exception as exc:
    generate_score = get_device = load_model = predict = preprocess_image = None
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
        except Exception as exc:
            inference_error = exc

    image_stack = ft.Stack(width=400, height=400)
    result_text = ft.Text(
        value="Choose an image to begin.",
        size=20,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )
    file_picker = ft.FilePicker()

    def show_image(source, label_text=""):
        controls = [
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
                border=ft.Border.all(width=4, color=ft.Colors.RED),
            ),
        ]
        if label_text:
            controls.append(
                ft.Container(
                    width=400,
                    height=400,
                    content=ft.Text(label_text, color=ft.Colors.WHITE, size=18),
                    alignment=ft.Alignment.BOTTOM_LEFT,
                    padding=ft.Padding.all(8),
                )
            )
        image_stack.controls = controls

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

        show_image(image_source)

        if inference_error is not None:
            result_text.value = f"Image loaded. Inference is unavailable: {inference_error}"
        elif not selected.path:
            result_text.value = "Image loaded, but inference requires a local file path."
        else:
            try:
                input_tensor = preprocess_image(str(selected.path), device)
                probs = predict(model, input_tensor)
                score = generate_score(probs)
                pred_idx = probs.argmax().item()
                pred_name = class_names[pred_idx] if class_names else f"class_{pred_idx}"
                show_image(image_source, f"{pred_name} ({score}%)")
                result_text.value = f"Predicted: {pred_name} – Confidence: {score}%"
            except Exception as exc:
                result_text.value = f"Inference error: {exc}"

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
                controls=[upload_btn, image_stack, result_text],
            ),
        )
    )


ft.run(main)
