from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from  env import load_env
from embeddings import  load_vector_db, load_embeddings
from llm import create_llm
from pipeline  import build_pipeline


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    referenced_pages: List[int]

# Global variable to hold the pipeline
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Code before yield runs on startup, code after yield runs on shutdown.
    """
    global rag_chain
    
    # Startup
    try:
        print("Initializing RAG Pipeline...")
        env = load_env()
        embeddings = load_embeddings(env)
        vectordb = load_vector_db(env, embeddings)
        llm = create_llm(env)
        
        # This pipeline now returns {"answer": str, "sources": List[Docs]}
        rag_chain = build_pipeline(vectordb, llm)
        print("RAG Pipeline ready.")
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        raise e
    
    yield  # Server is running
    
    print("Shutting down RAG Pipeline...")
    rag_chain = None

# --- App Initialization ---
app = FastAPI(
    title="RAG API", 
    description="Pilot Handbook QA System",
    lifespan=lifespan
)




@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Invoke the chain
        result = rag_chain.invoke(request.question)
        
        answer_text = result.get("answer", "No answer generated.")
        source_docs = result.get("sources", [])
        
        # Extract page numbers and remove duplicates
        # We use a set to handle uniqueness, then convert to sorted list
        unique_pages = set()
        for doc in source_docs:
            # Handle cases where page_number might be missing or None
            page_num = doc.metadata.get("page_number")
            if page_num is not None:
                # Ensure it's an integer
                try:
                    unique_pages.add(int(page_num))
                except (ValueError, TypeError):
                    pass # Skip invalid page numbers
        
        return QueryResponse(
            answer=answer_text,
            referenced_pages=sorted(list(unique_pages))
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


