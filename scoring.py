from utils import count_keyword_matches


def title_weighted_reranker(inputs):
    """
    Re-rank docs using:
      final_score = vector_score + (keyword_matches * weight)

    Inputs:
        inputs (dict):
            {
                "results": list[(Document, float)],
                "query": str,
                "weight": float
            }

    Returns:
        list[Document] (top 5)
    """
    results = inputs["results"]
    query = inputs["query"]
    weight = inputs["weight"]
    top_k = 5

    scored = []

    for doc, vector_score in results:
        title = doc.metadata.get("title", "")
        matches = count_keyword_matches(query, title)
        normalized=matches/len(query)
        boost = normalized * weight
        final_score = vector_score + boost
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
