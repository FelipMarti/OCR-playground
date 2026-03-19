from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

def run(image_path: str) -> str:
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words]
                text.append(" ".join(words))

    return "\n".join(text)
