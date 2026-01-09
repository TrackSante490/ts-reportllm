from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "../vectorstore/tracksante_chroma"

def get_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma(
        collection_name="tracksante_kb",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12}
    )
