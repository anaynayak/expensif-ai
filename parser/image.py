from typing import Iterator
from langchain_core.documents import Document

from model.clustering import cluster
from .vision import detect_text
from langchain_core.document_loaders import BaseBlobParser, Blob


class VisionImageParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        observations = VisionImageParser.parse(blob.source)
        text = "\n".join([observation.text for observation in observations])
        print(text)
        yield Document(
            page_content=text,
            metadata={"page": 1, "source": blob.source, "observations": observations},
        )

    @classmethod
    def parse(cls, path: str) -> Document:
        return cluster(path, detect_text(path))
