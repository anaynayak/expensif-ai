from typing import Iterator
from langchain_core.documents import Document

from model.clustering import cluster
from .vision import detect_text
from langchain_core.document_loaders import BaseBlobParser, Blob


class VisionImageParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        text = "\n".join(
            [observation.text for observation in cluster(detect_text(blob.source))]
        )
        print(text)
        yield Document(
            page_content=text,
            metadata={"page": 1, "source": blob.source},
        )
