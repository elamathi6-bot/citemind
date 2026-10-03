import os
import re
import json
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview"]
MAX_DISTANCE = 1.3  # refuse if the closest match is farther than this
TOP_K = 10
NUM_QUESTIONS = 5

embedder = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("course")


def label(m):
    if m["type"] == "video":
        mins, secs = divmod(m["timestamp"], 60)
        return f"Video {mins}:{secs:02d}"
    return f"Slide/Page {m['page']}"


def retrieve(question, k=TOP_K):
    q = embedder.encode([question]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def build_context(hits):
    return "\n\n".join(
        f"[{i+1}] ({label(m)})\n{doc}" for i, (doc, m, _) in enumerate(hits)
    )


def generate(prompt, json_mode=False):
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for model_name in [MODEL] + FALLBACK_MODELS:
        for attempt in range(3):
            try:
                resp = llm.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                print(f"(answered by {model_name})")
                return resp
            except Exception as e:
                msg = str(e)
                if "503" in msg or "429" in msg:
                    wait = 5 * (attempt + 1)
                    print(f"{model_name} busy, retrying in {wait}s...")
                    time.sleep(wait)
                elif "404" in msg:
                    print(f"{model_name} not available, trying next model...")
                    break
                else:
                    raise
    return None


# ---------------------------------------------------------------- Q&A

def answer(question):
    hits = retrieve(question)

    if not hits or hits[0][2] > MAX_DISTANCE:
        print("\nThis isn't covered in your course material.")
        if hits:
            print(f"(closest distance {hits[0][2]:.2f}, limit {MAX_DISTANCE})")
        return

    prompt = f"""You are a study tutor. Answer the student's question using ONLY the sources below.
Rules:
- Cite sources inline like [1] or [2] after each claim.
- Some sources are code or slides. Words may differ from the question (for example, the source may say "put" where the student says "insert"). Treat them as the same idea if the code or text clearly does that job.
- If a source contains code that answers the question, explain what it does step by step.
- Use only the sources that actually answer the question. Ignore loosely related ones.
- If the sources do not contain the answer, reply exactly: "This isn't covered in your course material."
- Do not use outside knowledge. Keep the answer short and clear.

SOURCES:
{build_context(hits)}

QUESTION: {question}"""

    resp = generate(prompt)
    if resp is None:
        print("All models are busy or unavailable right now. Try again in a few minutes.")
        return

    text = resp.text.strip()
    print("\n" + text)

    if "isn't covered in your course material" in text.lower():
        print(f"\n[debug] closest distance {hits[0][2]:.2f}. Chunks the model saw:")
        for i, (doc, m, dist) in enumerate(hits):
            preview = " ".join(doc.split())[:150]
            print(f"  [{i+1}] {label(m)} ({dist:.2f}): {preview}...")
        return

    # Show only the sources the answer actually cited
    cited = sorted({int(n) for grp in re.findall(r"\[([\d,\s]+)\]", text)
                    for n in re.findall(r"\d+", grp)})
    print("\nSources:")
    for n in cited:
        if 1 <= n <= len(hits):
            doc, m, dist = hits[n - 1]
            print(f"  [{n}] {label(m)}  (distance {dist:.2f})")


# ---------------------------------------------------------------- Quiz

def parse_quiz(text, num_sources):
    """Parse and validate the model's JSON. Returns a list of questions or None."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = data.get("questions", [])
    good = []
    for q in data:
        try:
            opts = q["options"]
            ans = str(q["answer"]).strip().upper()[:1]
            if len(opts) == 4 and ans in "ABCD" and q["question"] and q["explanation"]:
                src = int(q.get("source", 0))
                good.append({
                    "question": q["question"],
                    "options": opts,
                    "answer": ans,
                    "explanation": q["explanation"],
                    "source": src if 1 <= src <= num_sources else None,
                })
        except (KeyError, TypeError, ValueError):
            continue
    return good or None


def quiz(topic):
    hits = retrieve(topic)
    if not hits or hits[0][2] > MAX_DISTANCE:
        print("\nThis topic isn't covered in your course material.")
        return

    prompt = f"""You are a study tutor. Write {NUM_QUESTIONS} multiple-choice questions about "{topic}"
using ONLY the sources below. Do not use outside knowledge.

Rules:
- Each question has exactly 4 options and exactly one correct answer.
- Wrong options must be plausible but clearly wrong according to the sources.
- "source" is the number [n] of the source that supports the correct answer.
- Return ONLY a JSON list, with no other text, in this format:
[
  {{"question": "...", "options": ["...", "...", "...", "..."],
    "answer": "B", "explanation": "one or two sentences", "source": 3}}
]

SOURCES:
{build_context(hits)}"""

    questions = None
    for attempt in range(2):
        resp = generate(prompt, json_mode=True)
        if resp is None:
            print("All models are busy or unavailable right now. Try again in a few minutes.")
            return
        questions = parse_quiz(resp.text, len(hits))
        if questions:
            break
        print("Quiz came back malformed, trying once more...")
    if not questions:
        print("Could not generate a valid quiz. Try again or use a different topic.")
        return

    score = 0
    for n, q in enumerate(questions, 1):
        print(f"\nQ{n}. {q['question']}")
        for letter, opt in zip("ABCD", q["options"]):
            print(f"   {letter}) {opt}")
        choice = ""
        while choice not in list("ABCD"):
            choice = input("Your answer (A/B/C/D): ").strip().upper()[:1]
        if choice == q["answer"]:
            score += 1
            print("Correct!")
        else:
            print(f"Not quite. The answer is {q['answer']}.")
        print(f"   {q['explanation']}")
        if q["source"]:
            _, m, _ = hits[q["source"] - 1]
            print(f"   Source: {label(m)}")

    print(f"\nScore: {score}/{len(questions)}")


# ---------------------------------------------------------------- Main loop

if __name__ == "__main__":
    print("Ask any question about your course material.")
    print("Type 'quiz <topic>' for a quiz (example: quiz binary search tree), or 'quit' to exit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue
        if q.lower().startswith("quiz"):
            topic = q[4:].strip()
            if topic:
                quiz(topic)
            else:
                print("Add a topic, for example: quiz binary search tree")
        else:
            answer(q)