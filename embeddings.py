from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma


def load_embeddings(env):
    """
    Create embedding model for vector store.

    Inputs:
        env (dict): must contain key "HUGGING_FACE_TOKEN"

    Returns:
        HuggingFaceEndpointEmbeddings
    """
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=env["HUGGING_FACE_TOKEN"],
    )


def load_vector_db(env, embeddings):
    """
    Load persistent Chroma vector store.

    Inputs:
        env (dict): must contain "PERSIST_DIR"
        embeddings: embedding function instance

    Returns:
        Chroma vector database
    """
    return Chroma(
        persist_directory=env["PERSIST_DIR"],
        embedding_function=embeddings,
    )
