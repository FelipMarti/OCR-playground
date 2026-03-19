import sys
import os
import time
import json

from engines import plygrd_tesseract
from engines import plygrd_paddleocr
from engines import plygrd_doctr
from engines import plygrd_easyocr
from engines import plygrd_rapidocr
from engines import plygrd_trocr
from engines import plygrd_paddle_trocr


ENGINES = {
    "tesseract": plygrd_tesseract.run,
    "paddleocr": plygrd_paddleocr.run,
    "doctr": plygrd_doctr.run,
    "easyocr": plygrd_easyocr.run,
    "rapidocr": plygrd_rapidocr.run,
    "trocr": plygrd_trocr.run, 
    "paddle_trocr": plygrd_paddle_trocr.run,
}


def main(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    results = {}

    print(f"\nProcessing: {image_path}\n")

    for name, func in ENGINES.items():
        try:
            print(f"--- {name.upper()} ---")

            start = time.time()
            result = func(image_path)
            elapsed = time.time() - start

            results[name] = result

            if isinstance(result, dict):
                # Print raw OCR text
                text = result.get("text", "")
                print(text if text else "[EMPTY]")
            
                # Print structured JSON separately
                items = result.get("items", [])
                print(f"\n=== {name.upper()} JSON ===")
                print(json.dumps(items, indent=2))
            else:
                # Only text
                print(result if result else "[EMPTY]")
            
            print(f"[TIME]: {elapsed:.3f}s\n")

        except Exception as e:
            print(f"[ERROR in {name}]: {e}\n")
            results[name] = f"ERROR: {e}"

    # Write output.txt (overwrite)
    with open("output.txt", "w", encoding="utf-8") as f:
        for engine, result in results.items():
            if isinstance(result, dict):
                # Raw text section
                f.write(f"===== {engine.upper()} =====\n")
                f.write(result.get("text", "") + "\n\n")
    
                # JSON section
                f.write(f"=== {engine.upper()} JSON ===\n")
                f.write(json.dumps(result.get("items", []), indent=2) + "\n\n")
            else:
                # Old-style string engine
                f.write(f"===== {engine.upper()} =====\n")
                f.write(result + "\n\n")

    print("Results saved to output.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python OCR_playground.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])
