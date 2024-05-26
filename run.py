from langchain_community.document_loaders import PyPDFDirectoryLoader
from argparse import ArgumentParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from model.expense_report import model
from parser.image import VisionImageParser
from pprint import pprint as pp


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model", type=str, default="ollama/llama3")
    args = parser.parse_args()
    return args


def process(args):
    pdf_loader = PyPDFDirectoryLoader(args.data_path)
    image_loader = GenericLoader.from_filesystem(
        path=args.data_path,
        suffixes=[".png", ".jpg", ".jpeg"],
        show_progress=True,
        parser=VisionImageParser(),
    )
    loader_all = MergedDataLoader(loaders=[pdf_loader, image_loader])
    return loader_all.load()


if __name__ == "__main__":
    args = argparse()
    docs = process(args)
    for doc in docs:
        print(model(args.model, doc.metadata["source"], doc.page_content))
