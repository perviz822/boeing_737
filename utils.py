import re
import pandas as pd
import os

def clean_tokenize(text):
    """
    Normalize text by removing punctuation, converting to lowercase,
    and splitting into tokens.

    Args:
        text (str or None)

    Returns:
        list[str]
    """
    if not text:
        return []

    # remove punctuation
    text_no_punct = re.sub(r"[^\w\s]", "", text)

    # normalize case and split
    tokens = text_no_punct.lower().split()

    return tokens



def count_keyword_matches(query_text, title_text):
    """
    Count how many unique query words appear in the title.

    Args:
        query_text (str)
        title_text (str)

    Returns:
        int: number of matching unique words
    """
    query_words = set(clean_tokenize(query_text))
    title_words = set(clean_tokenize(title_text))

    matching_words = query_words.intersection(title_words)
    return len(matching_words)




def convert_tables_to_html(docs):
    """
    For each document marked as a table, load its CSV file and
    append an HTML representation of the table to the page content.

    Args:
        docs (list[Document])

    Returns:
        list[Document]: processed documents
    """
    processed_docs = []

    for doc in docs:
        is_table = doc.metadata.get("type") == "table"
        csv_path = doc.metadata.get("csv_path")

        if is_table and csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                html_table = df.to_html(index=False)
                doc.page_content += f"\n\n[TABLE_HTML]\n{html_table}"
            except Exception:
                print("There was an error parsing the content of the table to html for table",csv_path)
                pass

        processed_docs.append(doc)

    return processed_docs

