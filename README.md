### Expensif-ai

Uses Apple Vision APIs to extract text from images and processes the same using LLMs to flag any expenses that need a review by the finance team.

### Tech

1. Langchain
   1. Spring-boot equivalent for LLM integration.
   2. Useful utilities + integration for schema validation, retries etc
2. Langfuse
   1. LLMOps on local. `Justfile` provides recipe to download and start locally on docker
3. LiteLLM
   1. For easy switchover from local LLMs to Cloud hosted ones.
      1. Use `.envrc` + direnv to setup keys required
4. Gradio
   1. Lightweight user interface to demonstrate capabilities

### Background + Evolution

1. Langchain + easyocr
   1. Provided a working prototype but had extremely poor accuracy on thermal paper receipts
2. Langchain + pyobjc-framework-vision
   1. Improved the text extraction quality.
3. Langchain + PDF parsing
   1. Now removed but langchain makes PDF parsing easy as well.
4. Langchain schema enforcement
   1. Pydantic schema checks failed for all sorts of reasons, switching over to JSON schema worked better.
5. Easy Gemini testing by setting `GEMINI_API_KEY`
6. Langfuse for LLMOps
   1. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`

### Testing

Tests are written using `pytest`. A set of tests to run using the CORD dataset (https://github.com/clovaai/cord) is currently work in progress.


### Setup

Install `mise` , `just` and `docker`
1. `mise` if configured correctly should load up the right python version
2. `Justfile` provides a `just langfuse` command to run langfuse locally.
   1. Creds need to be setup manually in the env variables
3. `python run.py` launches a Gradio server which can be accessed on https://127.0.0.1:7860/
