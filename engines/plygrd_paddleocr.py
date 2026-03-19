from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def run(image_path: str) -> str:
    result = ocr.ocr(image_path)

    lines = []
    for line in result[0]:
        lines.append(line[1][0])

    return "\n".join(lines)
