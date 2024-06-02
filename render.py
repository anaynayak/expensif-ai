import os
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw
from observation import Observation
from ocrmac import ocrmac


def render(name: str, img_path: str, observations: List[Observation]):
    if not os.environ.get("RENDER"):
        return
    color = "red"
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)

    for observation in observations:
        x1, y1, x2, y2 = ocrmac.convert_coordinates_pil(
            (
                observation.bbox.x1,
                observation.bbox.y1,
                observation.bbox.width,
                observation.bbox.height,
            ),
            image.width,
            image.height,
        )

        draw.rectangle((x1, y1, x2, y2), outline=color)
        draw.text((x1, y2), observation.text, align="left", fill=color)

    image.save(f"/tmp/{Path(img_path).name}_{name}.png")
