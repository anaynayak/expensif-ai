from dataclasses import dataclass
from typing import List
from ocrmac import ocrmac


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def __repr__(self) -> str:
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_bbox(cls, x, y, width, height) -> "BoundingBox":
        return BoundingBox(x, y, x + width, y + height)

    def compute_iou(self, other: "BoundingBox") -> float:
        intersection_width = min(self.x2, other.x2) - max(self.x1, other.x1)
        intersection_height = min(self.y2, other.y2) - max(self.y1, other.y1)
        if intersection_width <= 0 or intersection_height <= 0:
            return 0

        intersection_area = intersection_width * intersection_height

        union_area = self.area + other.area - intersection_area
        iou = intersection_area / union_area
        return iou


@dataclass
class Observation:
    text: str
    confidence: float
    bbox: BoundingBox

    def __repr__(self) -> str:
        return f"DetectedText(text={self.text}, confidence={self.confidence}, bbox={self.bbox})"

    def __str__(self) -> str:
        return repr(self)

    def expand_till_left(self) -> "Observation":
        return Observation(
            self.text,
            self.confidence,
            BoundingBox(0, self.bbox.y1, 1, self.bbox.y2),
        )

    def overlaps(self, other: "Observation") -> bool:
        return self.bbox.compute_iou(other.bbox) > 0.2

    def merge(self, other: "Observation") -> "Observation":
        text = (
            f"{self.text} {other.text}"
            if self.bbox.x2 < other.bbox.x2
            else f"{other.text} {self.text}"
        )
        return Observation(
            text,
            (self.confidence + other.confidence) / 2,
            BoundingBox(
                min(self.bbox.x1, other.bbox.x1),
                min(self.bbox.y1, other.bbox.y1),
                max(self.bbox.x2, other.bbox.x2),
                max(self.bbox.y2, other.bbox.y2),
            ),
        )


def detect_text(img_path: str) -> List[Observation]:
    ocr_text = ocrmac.OCR(str(img_path))
    return [
        Observation(text, confidence, BoundingBox.from_bbox(x, y, w, h))
        for (text, confidence, (x, y, w, h)) in ocr_text.recognize()
    ]
