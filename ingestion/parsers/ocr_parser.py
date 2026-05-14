import os
import pytesseract

tesseract_path = os.environ.get("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from .base import BaseParser
from ingestion.schema import RawDocument


class OCRParser(BaseParser):
    def __init__(self, dpi: int = 300, lang: str = "eng", preprocess: bool = True):
        self.dpi = dpi
        self.lang = lang
        self.preprocess = preprocess

    def can_parse(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif")

    def parse(self, file_path: str) -> RawDocument:
        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            text = self._ocr_pdf(file_path)
            content_type = "pdf_scanned"
        else:
            text = self._ocr_image(file_path)
            content_type = "image"

        return RawDocument(
            source_path=file_path,
            content_type=content_type,
            raw_text=text,
            metadata={"ocr": True, "lang": self.lang, "preprocessed": self.preprocess}
        )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        # 1. Convert to grayscale
        image = image.convert("L")

        # 2. Deskew — find the angle text is tilted and rotate to correct it
        image = self._deskew(image)

        # 3. Increase contrast
        image = ImageEnhance.Contrast(image).enhance(2.0)

        # 4. Sharpen to make text edges crisper
        image = image.filter(ImageFilter.SHARPEN)

        # 5. Binarize (black & white) — Tesseract works best on pure B&W
        image = image.point(lambda x: 0 if x < 140 else 255, "1")

        return image

    def _deskew(self, image: Image.Image) -> Image.Image:
        """Detect skew angle using pixel projection and rotate to correct."""
        import cv2  # opencv-python

        img_array = np.array(image)

        # Threshold to binary
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find coordinates of all non-zero pixels (text)
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 10:
            return image  # not enough content to detect angle, skip

        # minAreaRect gives us the angle of the bounding box around all text
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle to the range (-45, 45)
        if angle < -45:
            angle = 90 + angle

        # Only rotate if skew is meaningful (ignore noise < 0.5 degrees)
        if abs(angle) < 0.5:
            return image

        return image.rotate(angle, expand=True, fillcolor=255)

    def _ocr_pdf(self, file_path: str) -> str:
        pages = convert_from_path(file_path, dpi=self.dpi)
        texts = []
        for page in pages:
            if self.preprocess:
                page = self._preprocess(page)
            texts.append(pytesseract.image_to_string(page, lang=self.lang))
        return "\n\n".join(texts)

    def _ocr_image(self, file_path: str) -> str:
        image = Image.open(file_path)
        if self.preprocess:
            image = self._preprocess(image)
        return pytesseract.image_to_string(image, lang=self.lang)