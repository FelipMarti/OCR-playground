from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import re


# ------------------------
# Load models 
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
    def get_key(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))
    return sorted(boxes, key=get_key)


def box_bounds(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def merge_boxes(boxes):
    """
    Merge multiple boxes into one big rectangle
    """
    x_min = min(box_bounds(b)[0] for b in boxes)
    y_min = min(box_bounds(b)[1] for b in boxes)
    x_max = max(box_bounds(b)[2] for b in boxes)
    y_max = max(box_bounds(b)[3] for b in boxes)

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def group_lines(boxes, y_threshold=12):
    """
    Group boxes that are on the same horizontal line
    """
    lines = []
    current_line = []

    for box in boxes:
        y = min(p[1] for p in box)

        if not current_line:
            current_line.append(box)
            continue

        prev_y = min(p[1] for p in current_line[-1])

        if abs(y - prev_y) < y_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    return lines


def crop_box(image, box, pad=5):
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]

    x_min = max(min(xs) - pad, 0)
    x_max = min(max(xs) + pad, image.width)
    y_min = max(min(ys) - pad, 0)
    y_max = min(max(ys) + pad, image.height)

    return image.crop((x_min, y_min, x_max, y_max))


def draw_boxes(image, boxes, save_path):
    draw = ImageDraw.Draw(image)

    for box in boxes:
        box = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(box, outline="red", width=2)

    image.save(save_path)


def trocr_batch(crops):
    pixel_values = processor(images=crops, return_tensors="pt", padding=True).pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return [
        t.strip() for t in decoded
        if isinstance(t, str) and t.strip()
    ]


def parse_line(line: str):
    """
    Extract structured data from a receipt line
    """
    pattern = r"^(\d+)\s+(.*?)\s+(\d+\.\d{2})\s+([A-Z])$"
    match = re.match(pattern, line)

    if not match:
        return None

    code, name, price, tax = match.groups()

    return {
        "code": code,
        "name": name,
        "price": float(price),
        "tax": tax
    }


def parse_receipt_metadata(lines):
    """
    Extract metadata like store name, total, GST, number of items, date/time
    """
    metadata = {}

    # Try to capture TOTAL
    for line in lines:
        if "TOTAL" in line.upper():
            m = re.search(r"\$?\s*([\d]+\.\d{2})", line)
            if m:
                metadata["total"] = float(m.group(1))
            break

    # Try to capture number of items
    for line in lines:
        m = re.search(r"(\d+)\s+ITEMS", line.upper())
        if m:
            metadata["num_items"] = int(m.group(1))
            break

    # Try to capture date / time
    for line in lines:
        m = re.search(r"(\d{2}\.\d{2}\.\d{2})\s+(\d{2}:\d{2})", line)
        if m:
            metadata["date"] = m.group(1)
            metadata["time"] = m.group(2)
            break

    # Capture store name (first line usually)
    if lines:
        metadata["store_name"] = lines[0]

    return metadata


# ------------------------
# Main entry
# ------------------------

def run(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Use working Paddle
    result = paddle_ocr.ocr(image_path)

    if result is None or len(result) == 0 or result[0] is None:
        return ""

    # Extract boxes
    boxes = [line[0] for line in result[0]]

    # Normalize
    boxes = [
        [[int(p[0]), int(p[1])] for p in box]
        for box in boxes
    ]

    print(f"[DEBUG] Raw boxes: {len(boxes)}")

    # Sort
    boxes = sort_boxes(boxes)

    os.makedirs("debug", exist_ok=True)
    draw_boxes(image.copy(), boxes, "debug/raw_boxes.jpg")

    # GROUP INTO LINES
    lines = group_lines(boxes)

    print(f"[DEBUG] Lines detected: {len(lines)}")

    # MERGE BOXES PER LINE
    merged_boxes = [merge_boxes(line) for line in lines]

    os.makedirs("debug", exist_ok=True)
    draw_boxes(image.copy(), merged_boxes, "debug/merged_lines.jpg")

    # Crop per line
    crops = [crop_box(image, box) for box in merged_boxes]

    # Batch TrOCR
    texts = trocr_batch(crops)

    # Parse text
    items = []
    others = []
    
    for line in texts:
        parsed = parse_line(line)
        if parsed:
            items.append(parsed)
        else:
            others.append(line)

    metadata = parse_receipt_metadata(texts)

    # Build full JSON
    receipt_json = {
        "metadata": metadata,
        "items": items,
        "others": others
    }

    return {
        "text": "\n".join(texts),
        "items": receipt_json,  
        "raw_lines": texts
    }
