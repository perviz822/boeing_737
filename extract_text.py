import fitz
import json

# ======================================================================
# ------------------------- HEADER DETECTION ----------------------------
# ======================================================================

def is_header(span_text: str, span: dict) -> bool:
    """
    Detect whether a piece of text represents a real section header.
    """
    correct_size = (span["size"] == 12.0)
    bold_font = ("Bold" in span["font"])
    not_np = not span_text.startswith("NP.")
    not_copyright = "Copyright" not in span_text

    return correct_size and bold_font and not_np and not_copyright


def collect_headers(doc, start_page: int, end_page: int):
    """
    Scan pages and collect all header spans.
    """
    headers = []

    for page_num in range(start_page, end_page + 1):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if is_header(text, span):
                        headers.append({
                            "page": page_num,
                            "y0": span["bbox"][1],  # vertical start position
                            "text": text
                        })

    return headers


def add_end_boundary(headers, end_page):
    """
    Add a boundary 'END' header to mark the end of the document.
    """
    headers.append({
        "page": end_page,
        "y0": float("inf"),
        "text": "END"
    })
    return headers


# ======================================================================
# ----------------------- TEXT EXTRACTION LOGIC -------------------------
# ======================================================================

def is_boilerplate(text: str) -> bool:
    """Detect unwanted footer/metadata lines to skip."""
    patterns = [
        text.startswith("Copyright © The Boeing Company"),
        text.startswith("D6-27370-TBC"),
        text.startswith("NP."),
        text == "August 30, 2000",
        text == "March 15, 2002",
    ]
    return any(patterns)


def extract_text_by_range(doc, start_header, end_header) -> str:
    """
    Extracts text based on the specific start and end coordinates.
    It determines the specific valid Y-range for the current page
    (Top-down for start page, Bottom-up for end page, Full for middle).
    """
    extracted_parts = []
    
    start_pg = start_header["page"]
    start_y = start_header["y0"]
    
    end_pg = end_header["page"]
    end_y = end_header["y0"]

    # Iterate only through the specific pages involved in this section
    for page_num in range(start_pg, end_pg + 1):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        # Define boundaries for the current page
        current_min_y = 0
        current_max_y = float('inf')

        if page_num == start_pg:
            current_min_y = start_y
        
        if page_num == end_pg:
            current_max_y = end_y

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    y = span["bbox"][1]
                    text = span["text"].strip()

                    # 1. Check if text is within the vertical boundaries of this page
                    # We use > and < strictly to avoid including the headers themselves
                    if current_min_y < y < current_max_y:
                        
                        # 2. Clean up and boilerplate check
                        if text and not is_boilerplate(text):
                            extracted_parts.append(text)

    return " ".join(extracted_parts).strip()


# ======================================================================
# --------------------------- MAIN FUNCTION -----------------------------
# ======================================================================

def extract_chunks_by_headers(
    pdf_path,
    start_page=2,
    end_page=4,
    save_to_json=False,
    output_file="texts_updated.json"
):
    """
    Extract chunks of text from the PDF using detected headers as boundaries.
    """

    doc = fitz.open(pdf_path)

    # Step 1 — Detect headers
    headers = collect_headers(doc, start_page, end_page)

    # Step 2 — Sort in reading order
    headers.sort(key=lambda h: (h["page"], h["y0"]))

    # Step 3 — Add    fake  final header for a sstop condition
    headers = add_end_boundary(headers, end_page)

    # Step 4 — Extract chunks
    chunks = []
    for i in range(len(headers) - 1):
        start = headers[i]
        end = headers[i + 1]

    
        section_text = extract_text_by_range(doc, start, end)

        if section_text:
            chunks.append({
                "section": start["text"],
                "page_number": start["page"] + 1,
                "description": section_text
            })

    # Optional: save result to JSON
    if save_to_json:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_file}")

    return chunks


# ======================================================================
# ---------------------------- EXAMPLE USE ------------------------------
# ======================================================================

if __name__ == "__main__":
    chunks = extract_chunks_by_headers(
        "raw_documents/boeing_manual.pdf",
        start_page=0, 
        end_page=145,
        save_to_json=True
    )

    for chunk in chunks:
        print("=" * 80)
        print(f"Section: {chunk['section']}")
        print(f"Page: {chunk['page_number']}")
        print("Text preview:")
        print(chunk['description'][:200] + "...")