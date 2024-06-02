from typing import List
from ocrmac import ocrmac

from observation import BoundingBox, Observation
from render import render


def detect_text(img_path: str) -> List[Observation]:
    ocr_text = ocrmac.OCR(str(img_path))
    observations = [
        Observation(text, confidence, BoundingBox.from_bbox(x, y, w, h))
        for (text, confidence, (x, y, w, h)) in ocr_text.recognize()
    ]
    render("vision", img_path, observations)
    return observations
