from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama


def load_db():
    loader = TextLoader("knowledge.txt")
    docs = loader.load()

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    return db


def get_explanation(db, query):
    docs = db.similarity_search(query)

    llm = ChatOllama(
        model="phi",
        base_url="http://127.0.0.1:11434"
    )

    response = llm.invoke(
        f"""
        You are an expert machine engineer.

        Based on the following information:
        {docs}

        Explain the reason for machine failure in 2-3 simple lines.
        Do NOT write code.
        Only give plain explanation.
        """
    )

    return response.content