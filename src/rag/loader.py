import os
import pdfplumber

# Loads PDFs + text files
def load_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            text = load_pdf(path)
        elif file.endswith(".txt"):
            text = load_txt(path)
        else:
            continue

        documents.append({
            "source": file,
            "content": text
        })

    return documents


# Splits large documents into chunks
def chunk_text(documents, chunk_size=50, overlap=10, max_chunks=20):
    chunks = []

    for doc in documents:
        text = doc["content"]
        start = 0

        while start < len(text) and len(chunks) < max_chunks:
            end = start + chunk_size

            chunks.append({
                "text": text[start:end],
                "source": doc["source"]
            })

            start += chunk_size - overlap

    return chunks