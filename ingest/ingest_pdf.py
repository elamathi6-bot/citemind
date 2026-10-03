import pymupdf as fitz
import chromadb
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/textbook.pdf"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

# start fresh each run so we never store duplicates
try:
    client.delete_collection("course")
except Exception:
    pass
col = client.get_or_create_collection("course")

def split_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def ingest():
    doc = fitz.open(PDF_PATH)
    ids, texts, metas = [], [], []
    for page_num, page in enumerate(doc, start=1):
        for i, chunk in enumerate(split_text(page.get_text())):
            if len(chunk.strip()) < 50:
                continue
            ids.append(f"pdf-p{page_num}-c{i}")
            texts.append(chunk)
            metas.append({"type": "pdf", "page": page_num})
    if not texts:
        print("No text found. The PDF may be scanned images.")
        return
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print(f"Stored {len(texts)} chunks from {len(doc)} pages")

def search(question, k=3):
    q = model.encode([question]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(f"\n[Page {meta['page']}]\n{doc[:300]}...")

if __name__ == "__main__":
    ingest()
    search("PUT A QUESTION YOUR CHAPTER ANSWERS HERE")