from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

from retrieval import retrieve_with_scores
from scoring import title_weighted_reranker
from utils import convert_tables_to_html


def build_pipeline(vectordb, llm):
    """
    Build the Retrieval-Augmented Generation (RAG) pipeline.
    The pipeline retrieves documents, re-ranks them using
    custom title-matching logic, enriches tables, and generates
    an answer along with the supporting sources.
    """

    # Prompt that the LLM receives.
    # It already includes a slot {context} for retrieved chunks
    # and {input} for the user question.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Use the retrieved context to answer the question. "
            "If the answer is unknown, say you don't know.\n\nContext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    # --------------------------------------------------------
    # STEP 1 — Attach retrieval parameters to the incoming query
    # --------------------------------------------------------
    # This stage:
    #   • passes the query unchanged
    #   • attaches the vectordb instance
    #   • defines how many documents to retrieve (k)
    #   • sets the weight for our custom title-match re-ranker
    #
    # Output example:
    #   {
    #       "query": "...",
    #       "vectordb": <Chroma instance>,
    #       "k": 20,
    #       "title_match_score_weight": 10
    #   }
    
    retrieval_inputs = RunnableParallel({
        "query": RunnablePassthrough(),
        "vectordb": lambda _: vectordb,
        "k": lambda _: 20,
        "title_match_score_weight": lambda _: 10,
    })

    # --------------------------------------------------------
    # STEP 2 — Retrieve + Custom Re-Ranking
    # --------------------------------------------------------
    # The pipeline now:
    #   1. Runs vector search (retrieve_with_scores)
    #   2. Feeds the raw vector results + query into our custom
    #      title_weighted_reranker
    #   3. The reranker boosts documents whose titles share
    #      important words with the query.
    #
    # Final output of this block:
    #   List[Document] — sorted by our combined score.
    retrieval_pipeline = (
        retrieval_inputs
        | RunnableParallel({
            "results": RunnableLambda(retrieve_with_scores),
            "query": itemgetter("query"),
            "weight": itemgetter("title_match_score_weight"),
        })
        | RunnableLambda(title_weighted_reranker)
    )

    # --------------------------------------------------------
    # STEP 3 — Enrich documents (table → HTML)
    # --------------------------------------------------------
    # If a chunk represents a table, we load its CSV, convert
    # it to HTML, and attach the HTML to the document content.
    # This lets the LLM “see” tables in a structured form.
    enriched_docs = retrieval_pipeline | convert_tables_to_html

    # --------------------------------------------------------
    # STEP 4 — Prepare final inputs for the LLM
    # --------------------------------------------------------
    # Build:
    #   {
    #       "context": <list of enriched docs>,
    #       "input":   <original query>
    #   }
    gather_stage = RunnableParallel({
        "context": enriched_docs,
        "input": RunnablePassthrough(),
    })

    # --------------------------------------------------------
    # STEP 5 — Generate the answer
    # --------------------------------------------------------
    # Prompt → LLM → string
    answer_chain = prompt | llm | StrOutputParser()

    # --------------------------------------------------------
    # STEP 6 — Final output
    # --------------------------------------------------------
    # Returns:
    #   {
    #       "answer": <generated answer>,
    #       "sources": <documents used as context>
    #   }
    return gather_stage | RunnableParallel({
        "answer": answer_chain,
        "sources": itemgetter("context"),
    })
