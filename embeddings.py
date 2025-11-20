
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path

def load_embeddings(env=None):
    """
    Load the all-mpnet-base-v2 embedding model and store/cache it
    in a 'models' folder located next to this embeddings.py file.

    Structure:
        embeddings.py
        models/
    """

    # Folder where this file lives (e.g. .../boeing_737/)
    module_dir = Path(__file__).resolve().parent

    # models/ next to embeddings.py
    model_dir = module_dir / "models"

    # Create the directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Tell HuggingFace to use this folder for caching models
    os.environ["HF_HOME"] = str(model_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}  
    )

    print(f"✔ Embedding model will be stored at: {model_dir}")
    return embeddings


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
