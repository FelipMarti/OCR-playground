from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

def run(image_path: str) -> str:
    result, _ = engine(image_path)

    if result is None:
        return ""

    lines = [res[1] for res in result]
    return "\n".join(lines)
