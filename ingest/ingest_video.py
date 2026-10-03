import chromadb
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

VIDEO_PATH = "data/lecture.mp4"
MAX_SECONDS = 600   # only first 10 min while testing; set to None for full video

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("course")

# remove old video chunks so reruns don't duplicate
try:
    col.delete(where={"type": "video"})
except Exception:
    pass

whisper = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = whisper.transcribe(VIDEO_PATH)

# group short segments into ~30-second chunks, keeping the start time
chunks, cur_text, cur_start = [], "", None
for seg in segments:
    if MAX_SECONDS and seg.start > MAX_SECONDS:
        break
    if cur_start is None:
        cur_start = seg.start
    cur_text += " " + seg.text.strip()
    if seg.end - cur_start >= 30:
        chunks.append((cur_start, cur_text.strip()))
        cur_text, cur_start = "", None
if cur_text.strip():
    chunks.append((cur_start, cur_text.strip()))

ids = [f"video-t{int(t)}" for t, _ in chunks]
texts = [txt for _, txt in chunks]
metas = [{"type": "video", "timestamp": int(t)} for t, _ in chunks]
col.add(ids=ids, documents=texts, metadatas=metas,
        embeddings=embedder.encode(texts).tolist())
print(f"Stored {len(texts)} video chunks")

q = embedder.encode(["how do you insert a node in a binary search tree"]).tolist()
res = col.query(query_embeddings=q, n_results=3, where={"type": "video"})
for doc, m in zip(res["documents"][0], res["metadatas"][0]):
    mins, secs = divmod(m["timestamp"], 60)
    print(f"\n[Video {mins}:{secs:02d}]\n{doc[:300]}...")