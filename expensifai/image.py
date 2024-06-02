from typing import Iterator, List, Tuple
from langchain_core.documents import Document

from expensifai.observation import Observation

from .clustering import cluster
from .vision import detect_text
from langchain_core.document_loaders import BaseBlobParser, Blob
from PIL.Image import Image


class VisionImageParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        observations, image = VisionImageParser.parse(blob.source)
        text = "\n".join([observation.text for observation in observations])
        print(text)
        yield Document(
            page_content=text,
            metadata={"page": 1, "source": blob.source, "observations": observations},
        )

    @classmethod
    def parse(cls, path: str) -> Tuple[List[Observation], Image]:
        observations, _ = detect_text(path)
        return cluster(path, observations)
