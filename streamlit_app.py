import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import io
import os
import pandas as pd

# ── document parsers ──────────────────────────────────────────────────────────
def parse_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")

def parse_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def parse_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string(index=False)

PARSERS = {
    "pdf":  parse_pdf,
    "txt":  parse_txt,
    "docx": parse_docx,
    "csv":  parse_csv,
}

# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]

# ── embedding + FAISS ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks: list[str]):
    embedder = get_embedder()
    vecs = embedder.encode(chunks, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, vecs

def retrieve(query: str, chunks: list[str], index, k: int = 4) -> list[str]:
    embedder = get_embedder()
    q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocChat 🧠", page_icon="🧠", layout="wide")

st.title("🧠 DocChat")
st.caption("Feed me your documents and ask me anything. I'll try not to make things up — no promises on the jokes though.")

# ── sidebar: API key + uploads ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    openai_api_key = st.text_input("OpenAI API Key", type="password",
                                   help="Get yours at platform.openai.com")

    st.divider()
    st.header("📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, DOCX or CSV",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # rebuild index only when file set changes
        file_key = tuple(sorted(f.name for f in uploaded_files))
        if st.session_state.get("_file_key") != file_key:
            with st.spinner("Reading & indexing your documents… 📚"):
                all_chunks = []
                for f in uploaded_files:
                    ext = f.name.rsplit(".", 1)[-1].lower()
                    parser = PARSERS.get(ext)
                    if parser:
                        text = parser(f.read())
                        all_chunks.extend(chunk_text(text))
                if all_chunks:
                    index, _ = build_index(all_chunks)
                    st.session_state["_chunks"]   = all_chunks
                    st.session_state["_index"]    = index
                    st.session_state["_file_key"] = file_key
                    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s)!")

    # show which docs are loaded
    if "_chunks" in st.session_state:
        st.info(f"📖 {len(st.session_state['_chunks'])} chunks ready to answer your questions.")
    else:
        st.warning("No documents loaded yet. I'll answer from memory (which is… questionable).")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── guard: need API key ───────────────────────────────────────────────────────
if not openai_api_key:
    st.info("👈 Pop your OpenAI API key in the sidebar to get started.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── build system prompt ───────────────────────────────────────────────────
    SYSTEM_BASE = (
        "You are a witty, slightly cheeky AI assistant with a talent for making "
        "even dry information entertaining. You use light humour, clever analogies, "
        "and the occasional pun — but never at the expense of accuracy. "
        "When you don't know something, admit it with flair instead of making things up. "
        "Keep answers focused and useful; the wit is seasoning, not the main dish."
    )

    if "_chunks" in st.session_state and "_index" in st.session_state:
        relevant = retrieve(prompt, st.session_state["_chunks"], st.session_state["_index"])
        context_block = "\n\n---\n\n".join(relevant)
        system_prompt = (
            f"{SYSTEM_BASE}\n\n"
            "You have been given relevant excerpts from the user's documents below. "
            "Base your answer primarily on these excerpts. "
            "If the excerpts don't cover the question, say so honestly.\n\n"
            f"=== DOCUMENT CONTEXT ===\n{context_block}\n=== END CONTEXT ==="
        )
    else:
        system_prompt = SYSTEM_BASE

    # ── call OpenAI ───────────────────────────────────────────────────────────
    messages_payload = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    stream = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages_payload,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
