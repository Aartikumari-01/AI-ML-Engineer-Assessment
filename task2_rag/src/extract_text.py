import pymupdf
import pytesseract
import json
import re

from PIL import Image
from pathlib import Path
from io import BytesIO


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
OUTPUT_FILE = PROJECT_DIR / "processed_pages.json"

ALLOWED_DOCUMENTS = [
    "UCI_AI4I_Documentation.pdf",
    "Predictive_Maintenance_IIoT_Research_2025.pdf"
]

all_pages = []

for document_name in ALLOWED_DOCUMENTS:

    pdf_file = DOCUMENTS_DIR / document_name

    if not pdf_file.exists():
        print(f"Document not found: {document_name}")
        continue

    print(f"\nReading: {pdf_file.name}")

    document = pymupdf.open(pdf_file)
    extracted_count = 0

    for page_number, page in enumerate(document, start=1):

        if (
            pdf_file.name == "UCI_AI4I_Documentation.pdf"
            and page_number == 4
        ):
            print(f"Skipping irrelevant UCI page: {page_number}")
            continue

        if (
            pdf_file.name == "Predictive_Maintenance_IIoT_Research_2025.pdf"
            and page_number >= 16
        ):
            print(f"Skipping reference page: {page_number}")
            continue

        page_text = page.get_text().strip()

        if not page_text:

            print(f"Using OCR for page: {page_number}")

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2)
            )

            image = Image.open(
                BytesIO(pix.tobytes("png"))
            )

            page_text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

        page_text = re.sub(
            r"(?im)^.*(?:acknowledge and accept the cookies|privacy practices|read policy|by using the uci machine learning repository|used by the uci machine learning repository).*$",
            "",
            page_text
        )

        page_text = re.sub(
            r"\n{2,}",
            "\n\n",
            page_text
        ).strip()

        if page_text:

            all_pages.append({
                "document": pdf_file.name,
                "page": page_number,
                "text": page_text
            })

            extracted_count += 1

    print(f"Pages with extracted text: {extracted_count}")

    document.close()


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_pages,
        file,
        ensure_ascii=False,
        indent=4
    )


print("\nPDF extraction completed!")
print("Total pages extracted:", len(all_pages))
print("Documents processed:", len(ALLOWED_DOCUMENTS))
print("Saved to:", OUTPUT_FILE)