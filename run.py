from langchain_community.document_loaders import PyPDFDirectoryLoader
from argparse import ArgumentParser


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model", type=str, default="ollama/llama3")
    args = parser.parse_args()
    return args


def process(args):
    loader = PyPDFDirectoryLoader(args.data_path)
    docs = loader.load()
    return docs


if __name__ == "__main__":
    args = argparse()
    docs = process(args)
    print(docs)
