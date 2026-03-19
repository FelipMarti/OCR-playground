# engines/plygrd_paddle_trocr.py

from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import torch
import numpy as np
import os

# ------------------------
# Load models ONCE
# ------------------------

paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)


# ------------------------
# Helpers
# ------------------------

def sort_boxes(boxes):
    """
    Sort boxes top-to-bottom, then left-to-right
    """
    def get_key(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))

    return sorted(boxes, key=get_key)


def crop_box(image, box):
    """
    Convert quadrilateral box → rectangular crop
    """
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]

    x_min, x_max = max(min(xs), 0), min(max(xs), image.width)
    y_min, y_max = max(min(ys), 0), min(max(ys), image.height)

    return image.crop((x_min, y_min, x_max, y_max))


def draw_boxes(image, boxes, save_path):
    draw = ImageDraw.Draw(image)

    for box in boxes:
        box = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(box, outline="red", width=2)

    image.save(save_path)


def trocr_recognize(image_crop):
    pixel_values = processor(images=image_crop, return_tensors="pt").pixel_values.to(device)

    generated_ids = trocr_model.generate(pixel_values)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    if not decoded or not isinstance(decoded[0], str):
        return ""
    
    return decoded[0].strip()


# ------------------------
# Main entry
# ------------------------

def run(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Use the WORKING Paddle call (with rec=True)
    result = paddle_ocr.ocr(image_path)

    if result is None or len(result) == 0 or result[0] is None:
        return ""

    # Extract boxes ONLY (ignore Paddle text)
    boxes = [line[0] for line in result[0]]

    print(f"[DEBUG] Detected {len(boxes)} boxes")

    # Normalize boxes → pure Python ints
    boxes = [
        [[int(p[0]), int(p[1])] for p in box]
        for box in boxes
    ]

    # Sort boxes
    boxes = sort_boxes(boxes)

    # Debug image
    os.makedirs("debug", exist_ok=True)
    draw_boxes(image.copy(), boxes, "debug/paddle_boxes.jpg")

    texts = []

    for box in boxes:
        crop = crop_box(image, box)

        try:
            text = trocr_recognize(crop)
        except Exception as e:
            print(f"[ERROR trocr] {e}")
            continue

        if isinstance(text, str):
            text = text.strip()
            if text:
                texts.append(text)

    return "\n".join(texts)
