import easyocr
from PIL import Image
import numpy as np

reader = easyocr.Reader(['en'], gpu=True)

def run(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")  
    img_np = np.array(img)

    results = reader.readtext(img_np)

    lines = [res[1] for res in results]
    return "\n".join(lines)
