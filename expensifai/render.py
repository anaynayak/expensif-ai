from typing import List
from PIL import Image, ImageDraw
from .observation import Observation
from ocrmac import ocrmac


def render(img_path: str, observations: List[Observation]) -> Image.Image:
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

    return image


def render_items(resp):
    return "\n".join(
        [
            f"<tr><td>{item['date']}</td><td>{item['name']}</td><td>{item['quantity']}</td><td>{item['amount']}</td><td>{item['category']}</td><td>{item['action']}</td></tr>"
            for item in resp["items"]
        ]
    )


def render_html(resp):
    return (
        """
<table>
<thead>
<tr>
<th>Date</th>
<th>Name</th>
<th>Quantity</th>
<th>Amount</th>
<th>Category</th>
<th>Action</th>
</tr>
</thead>
<tbody>"""
        + render_items(resp)
        + """
</tbody>
</table>
"""
    )
