import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from kb_loader import load_markdown_kb

PERSIST_DIR = "../vectordb/tracksante_chroma"

def build_vectordb():
    raw_docs = load_markdown_kb()
    if not raw_docs:
        raise RuntimeError("No knowledge base documents loaded.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(raw_docs)
    if not docs:
        raise RuntimeError("No non-empty chunks created. Check your markdown files.")

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    print(f"Loaded {len(raw_docs)} docs, split into {len(docs)} non-empty chunks")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="tracksante_kb"
    )

    vectordb.add_documents(docs)

    print(f"Indexed {len(docs)} chunks")

if __name__ == "__main__":
    build_vectordb()