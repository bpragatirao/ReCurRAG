"""
Document Loader Module for the RAG Pipeline.

Handles loading of three distinct data types:
1. Long Documents (PDFs) — arXiv research papers
2. Semi-Structured Data (CSVs) — Wine Quality dataset
3. Multi-Hop QA (JSON) — HotpotQA dataset

Each loader converts raw data into a unified list-of-dicts format:
    [{"source": <filename>, "content": <text_string>}, ...]
"""

import os
import json
import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------------
# Individual File Loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


def load_txt(file_path: str) -> str:
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_csv_as_text(file_path: str, separator: str = ";") -> str:
    """
    Convert a CSV file into a textual representation suitable for embedding.
    Each row becomes a natural-language sentence describing its feature values.
    """
    df = pd.read_csv(file_path, sep=separator)
    lines = []
    # Add schema description
    lines.append(f"Dataset columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Total records: {len(df)}")
    lines.append("")

    # Add summary statistics
    lines.append("Summary Statistics:")
    lines.append(df.describe().to_string())
    lines.append("")

    # Convert each row to natural language
    for idx, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items()]
        lines.append(f"Record {idx + 1}: " + ", ".join(parts))

    return "\n".join(lines)


def load_hotpotqa(file_path: str, max_samples: int = 500) -> list:
    """
    Load HotpotQA JSON and return a list of document dicts.

    Each question's context paragraphs are combined into a single document.
    We also preserve the question, answer, and supporting facts for
    evaluation purposes.

    Args:
        file_path: Path to the hotpotqa.json file.
        max_samples: Maximum number of QA pairs to load (the full file is ~90k).

    Returns:
        List of dicts with keys: source, content, question, answer,
        supporting_facts, level, type.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for i, item in enumerate(data[:max_samples]):
        # Build a single context string from all paragraphs
        context_parts = []
        for title, sentences in item.get("context", []):
            paragraph = " ".join(sentences)
            context_parts.append(f"[{title}]\n{paragraph}")

        context_text = "\n\n".join(context_parts)

        documents.append({
            "source": f"hotpotqa_sample_{i}",
            "content": context_text,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "supporting_facts": item.get("supporting_facts", []),
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })

    return documents


# ---------------------------------------------------------------------------
# Directory-Level Loaders
# ---------------------------------------------------------------------------

def load_documents(folder_path: str, data_type: str = "long_docs") -> list:
    """
    Load all documents from a folder based on data_type.

    Args:
        folder_path: Path to the data directory or file.
        data_type: One of 'long_docs', 'semi_structured', 'multi_hop'.

    Returns:
        List of dicts with at minimum 'source' and 'content' keys.
    """
    if data_type == "long_docs":
        return _load_long_docs(folder_path)
    elif data_type == "semi_structured":
        return _load_semi_structured(folder_path)
    elif data_type == "multi_hop":
        return _load_multi_hop(folder_path)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. "
                         f"Use 'long_docs', 'semi_structured', or 'multi_hop'.")


def _load_long_docs(folder_path: str) -> list:
    """Load PDFs and text files from a directory."""
    documents = []
    for file in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            text = load_pdf(path)
        elif file.endswith(".txt"):
            text = load_txt(path)
        else:
            continue

        if text.strip():
            documents.append({
                "source": file,
                "content": text
            })
            print(f"  Loaded: {file} ({len(text):,} chars)")

    return documents


def _load_semi_structured(folder_path: str) -> list:
    """Load CSV files from a directory and convert to text documents."""
    documents = []
    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".csv"):
            continue
        path = os.path.join(folder_path, file)
        text = load_csv_as_text(path, separator=";")
        if text.strip():
            documents.append({
                "source": file,
                "content": text
            })
            print(f"  Loaded: {file} ({len(text):,} chars)")

    return documents


def _load_multi_hop(file_path: str, max_samples: int = 500) -> list:
    """
    Load Multi-Hop QA data. file_path should point to the JSON file directly.
    """
    if os.path.isdir(file_path):
        # If given a directory, look for the JSON file inside
        json_files = [f for f in os.listdir(file_path) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {file_path}")
        file_path = os.path.join(file_path, json_files[0])

    docs = load_hotpotqa(file_path, max_samples=max_samples)
    print(f"  Loaded: {len(docs)} HotpotQA samples")
    return docs


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(documents: list, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split documents into overlapping chunks for embedding.

    Args:
        documents: List of dicts with 'source' and 'content' keys.
        chunk_size: Number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of dicts with 'text', 'source', and 'chunk_id' keys.
    """
    chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc["content"]
        source = doc["source"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text_content = text[start:end]

            # Only add non-trivial chunks
            if len(chunk_text_content.strip()) > 50:
                chunks.append({
                    "text": chunk_text_content,
                    "source": source,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

            start += chunk_size - overlap

    return chunks