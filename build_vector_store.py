import os
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  dotenv import load_dotenv
from embeddings import  load_embeddings

load_dotenv()


DATA_DIR = Path(".")  # change if your JSONs are in another folder
PERSIST_DIR = "chroma_db"  # local folder where Chroma will store data

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)


def load_json_docs(path: Path, default_type: str | None = None) -> list[Document]:
    """Load a JSON file of objects and return a list of embedded Documents."""
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    docs: list[Document] = []

    for obj in items:
        # Core fields
        obj_type = obj.get("type", default_type)
        page_number = obj.get("page_number")
        title = obj.get("title")
        description = obj.get("description")
        section = obj.get("section")
        csv_path = obj.get("csv_path")     # table CSV path if present

        # ------------------------------------------------------------------
        # 1. Load CSV text if this is a table and a CSV exists
        # ------------------------------------------------------------------
        table_text = ""
        if csv_path:
            csv_file = path.parent / csv_path
            if csv_file.exists():
                try:
                    with csv_file.open("r", encoding="utf-8") as f_csv:
                        table_text = f_csv.read()
                except Exception:
                    table_text = ""  # fail silently
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 2. Build the full embedded text: title + description + section + CSV
        # ------------------------------------------------------------------
        text_parts = []
        if title:
            text_parts.append(title)
        if description:
            text_parts.append(description)
        if section:
            text_parts.append(section)
        if table_text:
            text_parts.append(table_text)   
        # ------------------------------------------------------------------

        page_content = "\n\n".join(text_parts) if text_parts else ""

        # Metadata
        metadata = {
            "type": obj_type,
            "page_number": page_number,
            "title": title,
            "description": description,
            "section": section,
            "csv_path": csv_path,
        }

        # Create chunks
        chunks = splitter.create_documents(
            [page_content],
            metadatas=[metadata]
        )
        docs.extend(chunks)

    return docs



def build_vector_store():
    # 1) Load all documents from the three JSON files
    texts_path = DATA_DIR / "texts.json"
    tables_path = DATA_DIR / "tables.json"
    diagrams_path = DATA_DIR / "diagrams.json"

    all_docs: list[Document] = []
    all_docs += load_json_docs(texts_path, default_type="text")
    all_docs += load_json_docs(tables_path, default_type="table")
    all_docs += load_json_docs(diagrams_path, default_type="diagram")

    embeddings =load_embeddings()

    # 3) Create / overwrite local Chroma store
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    print(" Successfully indexed {len(all_docs)} documents into {PERSIST_DIR!r}")
    return vectordb


build_vector_store()