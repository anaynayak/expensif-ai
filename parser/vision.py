from typing import List, Optional, Tuple
from ocrmac import ocrmac


def detect_text(
    img_path: str, orientation: Optional[int] = None
) -> List[Tuple[str, float, List[float]]]:
    return ocrmac.OCR(str(img_path)).recognize()
