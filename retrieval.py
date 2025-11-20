

def retrieve_with_scores(inputs):
    """
    Perform similarity search with scores.

    Inputs:
        inputs (dict):
            {
                "query": str,
                "vectordb": Chroma,
                "k": int
            }

    Returns:
        list[(Document, float)]
    """
    query = inputs["query"]
    vectordb = inputs["vectordb"]
    k = inputs.get("k", 25)

    return vectordb.similarity_search_with_relevance_scores(query, k=k)




