from typing import List, Tuple
from ocrmac import ocrmac

from .observation import BoundingBox, Observation
from expensifai.render import render
from PIL.Image import Image


def detect_text(img_path: str) -> Tuple[List[Observation], Image]:
    ocr_text = ocrmac.OCR(str(img_path))
    observations = [
        Observation(text, confidence, BoundingBox.from_bbox(x, y, w, h))
        for (text, confidence, (x, y, w, h)) in ocr_text.recognize()
    ]
    return observations, render(img_path, observations)
