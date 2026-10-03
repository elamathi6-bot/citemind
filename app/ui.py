import os, re, json, time, hashlib, secrets, sqlite3, tempfile
import numpy as np
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from pptx import Presentation
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview"]
MAX_DISTANCE = 1.3
TOP_K = 10
REFUSAL = "This isn't covered in your course material."
VIDEO_EXT = {"mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a"}

st.set_page_config(page_title="Study Companion", page_icon="📚", layout="centered")

# ------------------------------------------------------------ look and feel

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, h1, h2, h3, h4 { font-family: 'Poppins', sans-serif; }
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
input, textarea { background:#F3F1FF !important; border-radius:10px !important; color:#1F1B3A !important; }
.stApp { background: linear-gradient(180deg, #F1EEFF 0%, #FFFFFF 40%); }
.hero { display:flex; align-items:center; gap:20px; padding:22px 26px; margin-bottom:18px;
  border-radius:22px; color:white;
  background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 55%, #74B9FF 100%);
  box-shadow: 0 10px 30px rgba(108,92,231,.35); }
.hero h1 { margin:0; font-size:1.9rem; font-weight:700; color:white; }
.hero p { margin:4px 0 0; opacity:.92; font-size:.95rem; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius:12px; border:0; font-weight:600; color:white;
  background: linear-gradient(135deg, #6C5CE7, #8E7CFF); }
.stButton > button:hover, .stFormSubmitButton > button:hover { filter:brightness(1.1); color:white; }
[data-baseweb="tab-list"] { gap:6px; }
[data-baseweb="tab"] { border-radius:12px 12px 0 0; font-weight:600; }
[data-testid="stExpander"] { border-radius:14px; background:white; box-shadow:0 2px 10px rgba(0,0,0,.05); }
[data-testid="stForm"] { border-radius:16px; background:white; box-shadow:0 2px 12px rgba(108,92,231,.10); }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

HERO_SVG = """<svg width="64" height="64" viewBox="0 0 64 64" fill="none">
<rect x="8" y="12" width="22" height="40" rx="4" fill="white" opacity=".95"/>
<rect x="34" y="12" width="22" height="40" rx="4" fill="white" opacity=".75"/>
<path d="M14 22h10M14 29h10M14 36h6M40 22h10M40 29h10M40 36h6" stroke="#6C5CE7" stroke-width="3" stroke-linecap="round"/>
<circle cx="52" cy="14" r="7" fill="#FFEAA7"/></svg>"""


def hero(sub):
    st.markdown(f'<div class="hero">{HERO_SVG}<div><h1>Study Companion</h1><p>{sub}</p></div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------ accounts and saved items

def db():
    con = sqlite3.connect("study.db")
    con.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT, pw TEXT)")
    con.execute("""CREATE TABLE IF NOT EXISTS saved(id INTEGER PRIMARY KEY AUTOINCREMENT,
                   username TEXT, kind TEXT, title TEXT, content TEXT,
                   created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    con.execute("""CREATE TABLE IF NOT EXISTS mastery(username TEXT, topic TEXT, p_known REAL,
                   attempts INTEGER, correct INTEGER, PRIMARY KEY (username, topic))""")
    con.execute("""CREATE TABLE IF NOT EXISTS asked(id INTEGER PRIMARY KEY AUTOINCREMENT,
                   username TEXT, question TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS topics(username TEXT, topic TEXT, prereqs TEXT,
                   PRIMARY KEY (username, topic))""")
    return con


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def signup(user, pw):
    user = user.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,20}", user):
        return "Username: 3 to 20 letters, numbers or underscores."
    if len(pw) < 6:
        return "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    try:
        with db() as con:
            con.execute("INSERT INTO users VALUES (?,?,?)", (user, salt, hash_pw(pw, salt)))
    except sqlite3.IntegrityError:
        return "That username is taken."
    return None


def login(user, pw):
    user = user.strip().lower()
    row = db().execute("SELECT salt, pw FROM users WHERE username=?", (user,)).fetchone()
    return bool(row) and secrets.compare_digest(row[1], hash_pw(pw, row[0]))


def save_item(kind, title, content):
    with db() as con:
        con.execute("INSERT INTO saved(username, kind, title, content) VALUES (?,?,?,?)",
                    (st.session_state.user, kind, title, content))
    st.toast("Saved to your library ✅")


# ------------------------------------------------------------ learner model (Bayesian knowledge tracing)

# Hand-set defaults, not fitted to data. State this honestly in the documentation.
P_INIT, P_TRANSIT, P_SLIP, P_GUESS = 0.30, 0.15, 0.10, 0.25


def bkt_update(p, correct):
    if correct:
        post = p * (1 - P_SLIP) / (p * (1 - P_SLIP) + (1 - p) * P_GUESS)
    else:
        post = p * P_SLIP / (p * P_SLIP + (1 - p) * (1 - P_GUESS))
    return post + (1 - post) * P_TRANSIT


def get_mastery():
    return db().execute("SELECT topic, p_known, attempts, correct FROM mastery "
                        "WHERE username=? ORDER BY p_known", (st.session_state.user,)).fetchall()


def canonical_concept(name):
    """Merge near-duplicate concept names so mastery isn't split across them."""
    name = name.strip().lower()
    existing = [r[0] for r in get_mastery()]
    if not existing:
        return name
    vecs = embedder.encode(existing + [name])
    q, rest = vecs[-1], vecs[:-1]
    sims = rest @ q / (np.linalg.norm(rest, axis=1) * np.linalg.norm(q))
    i = int(sims.argmax())
    return existing[i] if sims[i] >= 0.8 else name


def record_answer(concept, correct):
    topic = canonical_concept(concept)
    user = st.session_state.user
    with db() as con:
        row = con.execute("SELECT p_known, attempts, correct FROM mastery WHERE username=? AND topic=?",
                          (user, topic)).fetchone()
        p, att, cor = row if row else (P_INIT, 0, 0)
        con.execute("INSERT OR REPLACE INTO mastery VALUES (?,?,?,?,?)",
                    (user, topic, bkt_update(p, correct), att + 1, cor + int(correct)))


# ------------------------------------------------------------ course outline (topics and prerequisites)

def get_topics():
    rows = db().execute("SELECT topic, prereqs FROM topics WHERE username=?",
                        (st.session_state.user,)).fetchall()
    return [(t, json.loads(p)) for t, p in rows]


def build_outline():
    """One LLM pass over a sample of the uploaded material to get topics and their prerequisites."""
    docs = col.get(include=["documents"])["documents"]
    if not docs:
        return "empty"
    step = max(1, len(docs) // 40)
    excerpts = "\n---\n".join(d[:600] for d in docs[::step][:40])
    text, _m = generate(f"""These are excerpts from a course. Identify 6 to 12 major topics, in teaching order.
For each topic list its prerequisites: names of OTHER topics from your own list that should be learned first.
Use short lowercase topic names. Return ONLY a JSON list in this format:
[{{"topic": "binary search trees", "prerequisites": ["recursion"]}}]

EXCERPTS:
{excerpts}""", json_mode=True)
    if text is None:
        return "busy"
    try:
        items = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip())
        names = {str(t["topic"]).strip().lower() for t in items if t.get("topic")}
        rows = []
        for t in items:
            name = str(t.get("topic", "")).strip().lower()
            if not name:
                continue
            pre = [str(p).strip().lower() for p in t.get("prerequisites", [])]
            pre = [p for p in pre if p in names and p != name]
            rows.append((st.session_state.user, name, json.dumps(pre)))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return "malformed"
    if not rows:
        return "malformed"
    with db() as con:
        con.execute("DELETE FROM topics WHERE username=?", (st.session_state.user,))
        con.executemany("INSERT OR REPLACE INTO topics VALUES (?,?,?)", rows)
    return "ok"


# ------------------------------------------------------------ AI helpers

@st.cache_resource
def load_resources():
    llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    return llm, embedder, client


llm, embedder, chroma_client = load_resources()


def label(m):
    src = m.get("source", "Course")
    if m.get("type") == "video":
        mins, secs = divmod(int(m["timestamp"]), 60)
        return f"{src} {mins}:{secs:02d}"
    return f"{src} p.{m.get('page', '?')}"


def retrieve(question, k=TOP_K):
    if col.count() == 0:
        return []
    q = embedder.encode([question]).tolist()
    res = col.query(query_embeddings=q, n_results=min(k, col.count()))
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def build_context(hits):
    return "\n\n".join(f"[{i+1}] ({label(m)})\n{doc}" for i, (doc, m, _) in enumerate(hits))


def generate(contents, json_mode=False, models=None):
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for model_name in (models or [MODEL] + FALLBACK_MODELS):
        for attempt in range(3):
            try:
                resp = llm.models.generate_content(model=model_name, contents=contents, config=config)
                return resp.text, model_name
            except Exception as e:
                msg = str(e)
                if "503" in msg or "429" in msg:
                    time.sleep(5 * (attempt + 1))
                elif "404" in msg:
                    break
                else:
                    raise
    return None, None


def split_text(text, size=1200):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


def store(chunks, source):
    if not chunks:
        return 0
    texts = [c[0] for c in chunks]
    metas = [{**m, "source": source} for _, m in chunks]
    ids = [hashlib.md5(f"{source}-{i}".encode()).hexdigest() for i in range(len(chunks))]
    col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embedder.encode(texts).tolist())
    return len(chunks)


def ingest_pdf(path, name):
    chunks, empty = [], 0
    for pno, page in enumerate(PdfReader(path).pages, 1):
        pieces = split_text(page.extract_text() or "")
        empty += not pieces
        chunks += [(p, {"type": "pdf", "page": pno}) for p in pieces]
    return store(chunks, name), empty


def ingest_pptx(path, name):
    chunks, empty = [], 0
    for sno, slide in enumerate(Presentation(path).slides, 1):
        parts = [s.text_frame.text for s in slide.shapes if s.has_text_frame]
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
        pieces = split_text(" ".join(parts))
        empty += not pieces
        chunks += [(p, {"type": "slide", "page": sno}) for p in pieces]
    return store(chunks, name), empty


def ingest_video(path, name):
    up = llm.files.upload(file=path)
    while up.state.name == "PROCESSING":
        time.sleep(3)
        up = llm.files.get(name=up.name)
    if up.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini could not process this file (state {up.state.name}).")
    prompt = """Transcribe the speech in this recording.
Return ONLY a JSON list of segments of roughly 30 to 60 seconds each:
[{"start": <start time in whole seconds>, "text": "<what was said>"}]
Use only what is actually said. Do not summarize or add anything."""
    try:
        text, _ = generate([up, prompt], json_mode=True)
    finally:
        try:
            llm.files.delete(name=up.name)
        except Exception:
            pass
    if text is None:
        raise RuntimeError("All models are busy. Try again in a few minutes.")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    chunks = []
    for seg in json.loads(text):
        try:
            body = " ".join(str(seg["text"]).split())
            if body:
                chunks.append((body, {"type": "video", "timestamp": int(float(seg["start"]))}))
        except (KeyError, TypeError, ValueError):
            continue
    return store(chunks, name), 0


def list_sources():
    counts = {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        s = m.get("source", "Course")
        counts[s] = counts.get(s, 0) + 1
    return counts


def grounded(task, hits):
    text, model = generate(f"""You are a study tutor. {task}
Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- Cite sources inline like [1] or [2] after each claim.
- Sources may use different words than the question (for example "put" vs "insert"). Treat them as the same idea if the text or code clearly does that job.
- If a source contains code that answers the question, explain it step by step.
- Ignore loosely related sources. Keep it short and clear.
- If the sources do not contain the answer, reply exactly: "{REFUSAL}"

SOURCES:
{build_context(hits)}""")
    if text is None:
        return None, [], None
    text = text.strip()
    if "isn't covered in your course material" in text.lower():
        return REFUSAL, [], model
    cited = sorted({int(n) for g in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", g)})
    return text, [(n, *hits[n - 1]) for n in cited if 1 <= n <= len(hits)], model


def ask(question):
    hits = retrieve(question)
    if not hits or hits[0][2] > MAX_DISTANCE:
        return REFUSAL, [], None
    return grounded(f"Answer the student's question: {question}", hits)


def summarize(topic):
    hits = retrieve(topic, k=12)
    if not hits or hits[0][2] > MAX_DISTANCE:
        return REFUSAL, [], None
    return grounded(f'Write a study summary of "{topic}" as 5 to 8 short bullet points.', hits)


def parse_quiz(text, num_sources):
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
            ans = str(q["answer"]).strip().upper()[:1]
            if len(q["options"]) == 4 and ans in "ABCD" and q["question"] and q["explanation"]:
                src = int(q.get("source", 0))
                good.append({"question": q["question"], "options": q["options"], "answer": ans,
                             "explanation": q["explanation"],
                             "concept": str(q.get("concept") or "").strip() or None,
                             "source": src if 1 <= src <= num_sources else None})
        except (KeyError, TypeError, ValueError):
            continue
    return good or None


def past_questions(limit=50):
    rows = db().execute("SELECT question FROM asked WHERE username=? ORDER BY id DESC LIMIT ?",
                        (st.session_state.user, limit)).fetchall()
    return [r[0] for r in rows]


def remember_questions(questions):
    with db() as con:
        con.executemany("INSERT INTO asked(username, question) VALUES (?,?)",
                        [(st.session_state.user, q["question"]) for q in questions])


def is_repeat(question, others, threshold=0.85):
    """True if the question is very similar to one already asked."""
    if not others:
        return False
    vecs = embedder.encode(others + [question])
    q, rest = vecs[-1], vecs[:-1]
    sims = rest @ q / (np.linalg.norm(rest, axis=1) * np.linalg.norm(q))
    return float(sims.max()) >= threshold


def verify_quiz(questions, hits):
    """Cross-model check: a different model answers blind; keep only questions where it agrees with the key."""
    if not questions:
        return [], False
    blind = [{"id": i, "question": q["question"], "options": q["options"]} for i, q in enumerate(questions)]
    text, _m = generate(f"""Answer each question using ONLY the sources below.
Return ONLY a JSON list like [{{"id": 0, "answer": "B"}}], one entry per question.

SOURCES:
{build_context(hits)}

QUESTIONS:
{json.dumps(blind)}""", json_mode=True, models=FALLBACK_MODELS + [MODEL])
    if text is None:
        return questions, False
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip())
        agreed = {int(d["id"]) for d in data
                  if str(d["answer"]).strip().upper()[:1] == questions[int(d["id"])]["answer"]}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        return questions, False
    return [q for i, q in enumerate(questions) if i in agreed], True


def make_quiz(topic, n, difficulty):
    hits = retrieve(topic)
    if not hits or hits[0][2] > MAX_DISTANCE:
        return None, hits, "not_covered"
    past = past_questions(50)
    names = [t for t, _ in get_topics()]
    concept_rule = (f'- "concept" must be exactly one of these course topics: {names}' if names else
                    '- "concept" is a 2 to 4 word name for the specific idea the question tests.')
    prompt = f"""You are a study tutor. Write {n + 2} multiple-choice questions about "{topic}"
using ONLY the sources below. Do not use outside knowledge. Difficulty: {difficulty}.
Rules:
- Each question has exactly 4 options and exactly one correct answer.
- Wrong options must be plausible but clearly wrong according to the sources.
- Mix question types: facts, how a process or code works, and why something is done.
- Do not ask two questions that test the same fact.
- Do not repeat or closely paraphrase these earlier questions: {past[:15]}
- "source" is the number [n] of the source that supports the correct answer.
{concept_rule}
- Return ONLY a JSON list, no other text, in this format:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": "B", "explanation": "one or two sentences", "source": 3, "concept": "bst insertion"}}]

SOURCES:
{build_context(hits)}"""
    for _ in range(2):
        text, _m = generate(prompt, json_mode=True)
        if text is None:
            return None, hits, "busy"
        questions = parse_quiz(text, len(hits))
        if questions:
            drafted = len(questions)
            kept = []
            for q in questions:
                if not is_repeat(q["question"], past + [k["question"] for k in kept]):
                    kept.append(q)
            kept, checked = verify_quiz(kept, hits)
            kept = kept[:n]
            if kept:
                remember_questions(kept)
                note = (f"{len(kept)} of {drafted} drafted questions kept "
                        f"(repeats removed; answer keys cross-checked by a second model)." if checked else
                        f"{len(kept)} questions kept (repeats removed; answer keys could not be cross-checked this time).")
                st.session_state["quiz_note"] = note
                return kept, hits, "ok"
    return None, hits, "malformed"


def make_diagnostic(max_topics=8):
    """One question per course topic from a single model call, to place a new student quickly."""
    names = [t for t, _ in get_topics()][:max_topics]
    if not names:
        return None, [], "no_outline"
    hits, seen = [], set()
    for name in names:
        for h in retrieve(name, k=2):
            if h[0] not in seen:
                seen.add(h[0])
                hits.append(h)
    if not hits:
        return None, [], "not_covered"
    past = past_questions(50)
    prompt = f"""You are a study tutor. Write one multiple-choice question for each of these course topics: {names}.
Use ONLY the sources below. Do not use outside knowledge. Difficulty: medium.
Rules:
- Each question has exactly 4 options and exactly one correct answer.
- Wrong options must be plausible but clearly wrong according to the sources.
- "concept" must be exactly the topic name the question belongs to.
- "source" is the number [n] of the source that supports the correct answer.
- Skip a topic if the sources do not cover it.
- Do not repeat or closely paraphrase these earlier questions: {past[:15]}
- Return ONLY a JSON list, no other text, in this format:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": "B", "explanation": "one or two sentences", "source": 3, "concept": "{names[0]}"}}]

SOURCES:
{build_context(hits)}"""
    for _ in range(2):
        text, _m = generate(prompt, json_mode=True)
        if text is None:
            return None, hits, "busy"
        questions = parse_quiz(text, len(hits))
        if questions:
            kept, checked = verify_quiz(questions, hits)
            if kept:
                remember_questions(kept)
                st.session_state["quiz_note"] = (
                    f"Diagnostic: {len(kept)} questions across {len(names)} topics"
                    + ("; answer keys cross-checked by a second model." if checked
                       else "; answer keys could not be cross-checked this time."))
                return kept, hits, "ok"
    return None, hits, "malformed"


def run_and_store(key, title, fn, *args):
    st.session_state[key] = (title, *fn(*args))


def render_result(key, kind):
    r = st.session_state.get(key)
    if not r:
        return
    title, text, sources, model = r
    if text is None:
        st.error("All models are busy right now. Try again in a few minutes.")
        return
    if text == REFUSAL:
        st.warning(REFUSAL)
        return
    st.markdown(text)
    st.subheader("Sources")
    for n, doc, m, dist in sources:
        with st.expander(f"[{n}] {label(m)}  (distance {dist:.2f})"):
            st.text(doc)
    if model:
        st.caption(f"Answered by {model}")
    content = text + "\n\nSources:\n" + "\n".join(f"[{n}] {label(m)}" for n, _, m, _ in sources)
    st.button("💾 Save this", key=f"save_{key}", on_click=save_item, args=(kind, title, content))


# ------------------------------------------------------------ login gate

if "user" not in st.session_state:
    hero("Sign in to keep your material and saved notes private.")
    t_in, t_up = st.tabs(["Log in", "Sign up"])
    with t_in:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                if login(u, p):
                    st.session_state.user = u.strip().lower()
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
    with t_up:
        with st.form("signup_form"):
            u2 = st.text_input("Choose a username")
            p2 = st.text_input("Choose a password", type="password")
            if st.form_submit_button("Create account"):
                err = signup(u2, p2)
                if err:
                    st.error(err)
                else:
                    st.session_state.user = u2.strip().lower()
                    st.rerun()
    st.stop()

USER = st.session_state.user
col = chroma_client.get_or_create_collection("u_" + hashlib.md5(USER.encode()).hexdigest()[:16])

with st.sidebar:
    st.markdown(f"### 👋 {USER}")
    st.caption("Your uploads and saved items are private to this account.")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

hero("Upload your course material. Answers come only from it, with citations.")

tab_up, tab_ask, tab_sum, tab_quiz, tab_map, tab_prog, tab_saved = st.tabs(
    ["📤 Upload", "💬 Ask", "📝 Summary", "🎯 Quiz", "🗺️ Topics", "📈 Progress", "⭐ Saved"]
)

# ---- Upload
with tab_up:
    files = st.file_uploader("PDF, PowerPoint (.pptx), or video/audio",
                             type=["pdf", "pptx"] + sorted(VIDEO_EXT), accept_multiple_files=True)
    if st.button("Add to my course", disabled=not files):
        for f in files:
            ext = f.name.rsplit(".", 1)[-1].lower()
            with st.status(f"Processing {f.name}...", expanded=True) as status:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + ext)
                try:
                    tmp.write(f.getbuffer())
                    tmp.close()
                    if ext == "pdf":
                        n, empty = ingest_pdf(tmp.name, f.name)
                    elif ext == "pptx":
                        n, empty = ingest_pptx(tmp.name, f.name)
                    else:
                        st.write("Transcribing with Gemini. Long videos can take a few minutes.")
                        n, empty = ingest_video(tmp.name, f.name)
                    msg = f"Stored {n} chunks."
                    if empty:
                        msg += f" {empty} pages had no readable text (scanned images?)."
                    status.update(label=f"{f.name}: {msg}", state="complete")
                except Exception as e:
                    status.update(label=f"{f.name} failed: {e}", state="error")
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
    st.divider()
    st.subheader("Your course material")
    sources = list_sources()
    if not sources:
        st.info("Nothing uploaded yet.")
    for s, count in sources.items():
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{s}**: {count} chunks")
        if c2.button("Delete", key=f"del_{s}"):
            col.delete(where={"source": s})
            st.rerun()
    if sources and st.button("Clear everything"):
        chroma_client.delete_collection(col.name)
        st.rerun()

# ---- Ask
with tab_ask:
    with st.form("ask_form"):
        question = st.text_input("Your question", placeholder="What are the rules of a binary search tree?")
        go = st.form_submit_button("Ask")
    if go and question.strip():
        if col.count() == 0:
            st.warning("Upload some material first.")
        else:
            with st.spinner("Searching your material..."):
                run_and_store("res_ask", question.strip(), ask, question.strip())
    render_result("res_ask", "Answer")

# ---- Summary
with tab_sum:
    with st.form("sum_form"):
        sum_topic = st.text_input("Topic to summarize", placeholder="binary search tree")
        go_sum = st.form_submit_button("Summarize")
    if go_sum and sum_topic.strip():
        if col.count() == 0:
            st.warning("Upload some material first.")
        else:
            with st.spinner("Writing summary..."):
                run_and_store("res_sum", sum_topic.strip(), summarize, sum_topic.strip())
    render_result("res_sum", "Summary")

# ---- Quiz
with tab_quiz:
    have_topics = bool(get_topics())
    if not have_topics:
        st.info("Tip: build the course outline in the Topics tab to unlock a diagnostic quiz "
                "and topic-based mastery tracking.")
    elif not get_mastery():
        st.info("New here? Take the diagnostic quiz so the app can see where you stand.")
    if have_topics and st.button("🧭 Diagnostic quiz (one question per topic)", disabled=col.count() == 0):
        with st.spinner("Writing a question for each topic..."):
            questions, hits, status = make_diagnostic()
        if status == "ok":
            st.session_state.qid = st.session_state.get("qid", 0) + 1
            st.session_state.quiz = {"questions": questions, "hits": hits, "done": False,
                                     "topic": "diagnostic",
                                     "note": st.session_state.get("quiz_note", "")}
        else:
            st.session_state.pop("quiz", None)
            st.warning({"no_outline": "Build the course outline first (Topics tab).",
                        "not_covered": "Your material doesn't cover these topics.",
                        "busy": "All models are busy right now. Try again in a few minutes.",
                        "malformed": "Could not generate a valid diagnostic. Try again."}[status])
    with st.form("topic_form"):
        topic = st.text_input("Quiz topic", placeholder="binary search tree")
        c1, c2 = st.columns(2)
        num_q = c1.slider("Number of questions", 3, 10, 5)
        diff = c2.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        make = st.form_submit_button("Generate quiz")
    if make and topic.strip():
        if col.count() == 0:
            st.warning("Upload some material first.")
        else:
            with st.spinner("Writing questions..."):
                questions, hits, status = make_quiz(topic.strip(), num_q, diff)
            if status == "ok":
                st.session_state.qid = st.session_state.get("qid", 0) + 1
                st.session_state.quiz = {"questions": questions, "hits": hits, "done": False,
                                         "topic": topic.strip(),
                                         "note": st.session_state.get("quiz_note", "")}
            else:
                st.session_state.pop("quiz", None)
                st.warning({"not_covered": "This topic isn't covered in your course material.",
                            "busy": "All models are busy right now. Try again in a few minutes.",
                            "malformed": "Could not generate a valid quiz. Try again or pick another topic."}[status])

    quiz = st.session_state.get("quiz")
    if quiz:
        questions, hits, qid = quiz["questions"], quiz["hits"], st.session_state.qid
        if quiz.get("note"):
            st.caption(quiz["note"])
        with st.form(f"quiz_form_{qid}"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                st.radio("Choose one", [f"{L}) {o}" for L, o in zip("ABCD", q["options"])],
                         index=None, key=f"q{qid}_{i}", label_visibility="collapsed")
            if st.form_submit_button("Submit answers"):
                quiz["done"] = True
        if quiz["done"]:
            first_time = not quiz.get("recorded")  # record mastery only once per quiz
            score, lines, results = 0, [], []
            for i, q in enumerate(questions):
                picked = st.session_state.get(f"q{qid}_{i}")
                ok = bool(picked) and picked[0] == q["answer"]
                score += ok
                results.append((q, ok))
                if first_time and picked:
                    record_answer(q.get("concept") or quiz["topic"], ok)
                src = f" (Source: {label(hits[q['source'] - 1][1])})" if q["source"] else ""
                lines.append(f"Q{i+1}. {q['question']}\nCorrect answer: {q['answer']}. "
                             f"You chose: {picked[0] if picked else 'none'}.\n{q['explanation']}{src}\n")
                with st.expander(f"{'✅' if ok else '❌'} Q{i+1}: correct answer {q['answer']}"):
                    if picked and not ok:
                        st.write(f"You chose: {picked}")
                    st.write(q["explanation"])
                    if src:
                        st.caption(src.strip(" ()"))
            quiz["recorded"] = True
            st.success(f"Score: {score}/{len(questions)}")
            by_topic = {}
            for q, ok in results:
                name = q.get("concept") or quiz["topic"]
                right, total = by_topic.get(name, (0, 0))
                by_topic[name] = (right + int(ok), total + 1)
            st.subheader("Report")
            for name, (right, total) in by_topic.items():
                st.write(f"{'✅' if right == total else '⚠️'} **{name}**: {right}/{total} correct")
            wrong = [q for q, ok in results if not ok]
            if wrong:
                spots = sorted({label(hits[q["source"] - 1][1]) for q in wrong if q["source"]})
                if spots:
                    st.info("Review these parts of your material: " + "; ".join(spots))
            st.button("💾 Save quiz and results", key=f"save_quiz_{qid}", on_click=save_item,
                      args=("Quiz", f"Quiz: {quiz['topic']} ({score}/{len(questions)})",
                            f"Score: {score}/{len(questions)}\n\n" + "\n".join(lines)))

# ---- Topics
with tab_map:
    if st.button("Build / refresh course outline", disabled=col.count() == 0):
        with st.spinner("Reading your material and finding topics..."):
            res = build_outline()
        if res != "ok":
            st.warning({"empty": "Upload some material first.",
                        "busy": "All models are busy right now. Try again in a few minutes.",
                        "malformed": "Could not build a valid outline. Try again."}[res])
    topics = get_topics()
    if not topics:
        st.info("Upload your material, then build the course outline. Quiz questions will be tagged "
                "with these topics so your mastery builds up per topic.")
    else:
        mastery = {t: p for t, p, _, _ in get_mastery()}

        def node_colour(t):
            p = mastery.get(t)
            if p is None:
                return "#DFE6E9"
            return "#55EFC4" if p >= 0.7 else "#FFEAA7" if p >= 0.4 else "#FAB1A0"

        def q(s):
            return s.replace('"', "'")

        dot = 'digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica];'
        for t, pre in topics:
            dot += f' "{q(t)}" [fillcolor="{node_colour(t)}"];'
            for p in pre:
                dot += f' "{q(p)}" -> "{q(t)}";'
        dot += " }"
        st.graphviz_chart(dot)
        st.caption("Arrows point from a prerequisite to the topic that builds on it. "
                   "Green: mastered, yellow: in progress, red: weak, grey: not tested yet.")

# ---- Progress
with tab_prog:
    rows = get_mastery()
    if not rows:
        st.info("Take a quiz to start building your mastery profile.")
    else:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["Concept", "Mastery", "Attempts", "Correct"]).set_index("Concept")
        st.bar_chart(df["Mastery"])
        weak = [r for r in rows if r[1] < 0.6]
        if weak:
            st.warning("Weak concepts: " + ", ".join(r[0] for r in weak[:5]))
        shown = df.copy()
        shown["Mastery"] = (shown["Mastery"] * 100).round().astype(int).astype(str) + "%"
        st.dataframe(shown)

# ---- Saved
with tab_saved:
    rows = db().execute("SELECT id, kind, title, content, created FROM saved WHERE username=? "
                        "ORDER BY id DESC", (USER,)).fetchall()
    if not rows:
        st.info("Nothing saved yet. Use the 💾 Save button on an answer, summary or quiz.")
    for rid, kind, title, content, created in rows:
        with st.expander(f"{kind}: {title[:70]}  ·  {created[:16]}"):
            if kind == "Quiz":
                st.text(content)
            else:
                st.markdown(content)
            d1, d2 = st.columns(2)
            d1.download_button("Download", content, file_name=f"{kind}_{rid}.txt", key=f"dl_{rid}")
            if d2.button("Delete", key=f"rm_{rid}"):
                with db() as con:
                    con.execute("DELETE FROM saved WHERE id=? AND username=?", (rid, USER))
                st.rerun()