"""
Fordham RAG - Streamlit app. Ask questions about Fordham University.
All RAG logic imported from Lecture 5 notebook.
"""
import sys
from pathlib import Path

# Ensure helpers is on path
_lectures = Path(__file__).resolve().parent
if str(_lectures) not in sys.path:
    sys.path.insert(0, str(_lectures))

import streamlit as st
import zipfile
import numpy as np
from dotenv import load_dotenv
import litellm

from helpers import get_local_model, batch_embed_local, batch_embed_openai, batch_cosine_similarity

load_dotenv()

# ============ 1. Load docs (from Lecture 5 Cell 3) ============
def clean_page_name(path_str: str) -> str:
    fname = Path(path_str).name
    stem = Path(fname).stem
    return stem.replace("-", " ").replace("_", " ").strip().title()


@st.cache_data
def load_docs():
    zip_path = _lectures / "fordham-website.zip"
    if not zip_path.exists():
        zip_path = Path("fordham-website.zip")
    docs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            content_bytes = zf.read(info.filename)
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("latin-1", errors="ignore")
            page_name = clean_page_name(info.filename)
            docs.append({"page_name": page_name, "content": text})
    return docs


# ============ 2. Chunk documents (from Lecture 5 Cell 5) ============
def chunk_documents(docs, chunk_size=900, overlap=0):
    chunks = []
    for doc in docs:
        page_name = doc["page_name"]
        text = doc["content"]
        lines = text.strip().split("\n")
        source_url = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        i = 0
        chunk_index = 0
        while i < len(body):
            chunk_text = body[i : i + chunk_size]
            chunks.append({
                "page_name": page_name,
                "source_url": source_url,
                "chunk_index": chunk_index,
                "content": chunk_text,
            })
            chunk_index += 1
            i += chunk_size - overlap if overlap > 0 else chunk_size
    return chunks


# ============ 3. Embed (from Lecture 5 Cell 7) ============
EMBED_MODEL_TYPE = "openai"  # Use local for no API; set "openai" if you have embeddings
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"


def _truncate_for_openai(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def embed_texts(texts, model_type=EMBED_MODEL_TYPE, show_progress=True):
    if model_type == "local":
        return batch_embed_local(texts, model_name=LOCAL_MODEL_NAME, show_progress=show_progress)
    if model_type == "openai":
        safe_texts = [_truncate_for_openai(t) for t in texts]
        return batch_embed_openai(safe_texts, model=OPENAI_MODEL_NAME, batch_size=50, verbose=show_progress)
    raise ValueError(f"model_type must be 'local' or 'openai', got {model_type}")


def get_embedding_model(model_type=EMBED_MODEL_TYPE):
    if model_type == "local":
        return get_local_model(LOCAL_MODEL_NAME)
    return None


@st.cache_data
def load_chunks_and_embeddings():
    docs = load_docs()
    chunks = chunk_documents(docs)
    emb_path = _lectures / "chunk_embeddings.npy"
    if emb_path.exists():
        chunk_embeddings = np.load(emb_path)
        # Embeddings may be for subset; align chunks
        n = len(chunk_embeddings)
        chunks_to_embed = chunks[:n]
    else:
        texts = [c["content"] for c in chunks]
        chunk_embeddings = embed_texts(texts, model_type=EMBED_MODEL_TYPE, show_progress=True)
        chunks_to_embed = chunks
    return chunks_to_embed, chunk_embeddings


# ============ 4. Retrieve (from Lecture 5 Cell 9) ============
def retrieve_chunks(question, chunks_to_embed, chunk_embeddings, k=5, model_type=EMBED_MODEL_TYPE):
    if model_type == "local":
        model = get_embedding_model("local")
        query_emb = model.encode(question, convert_to_numpy=True)
    elif model_type == "openai":
        q_text = _truncate_for_openai(question)
        q_embs = batch_embed_openai([q_text], model=OPENAI_MODEL_NAME, batch_size=1, verbose=False)
        query_emb = q_embs[0]
    else:
        raise ValueError("model_type must be 'local' or 'openai'")
    scores = batch_cosine_similarity(query_emb, chunk_embeddings)
    top_idx = np.argsort(-scores)[:k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        chunk = chunks_to_embed[idx]
        results.append({"rank": rank, "score": float(scores[idx]), **chunk})
    return results


# ============ 5. Generate (from Lecture 5 Cell 12) ============
def generate_answer(question, retrieved_chunks, model="gpt-4o-mini"):
    context_parts = [f"[Source: {c.get('page_name', 'Unknown')}]\n{c['content']}" for c in retrieved_chunks]
    context = "\n\n---\n\n".join(context_parts)
    system = (
        "You are a helpful assistant that answers questions about Fordham University "
        "using only the provided context. If the context does not contain the answer, "
        "say so clearly. Do not make up information."
    )
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ============ 6. RAG (from Lecture 5 Cell 14) ============
def rag(question, chunks_to_embed, chunk_embeddings, k=5, model="gpt-4o-mini"):
    chunks = retrieve_chunks(question, chunks_to_embed, chunk_embeddings, k=k, model_type=EMBED_MODEL_TYPE)
    answer = generate_answer(question, chunks, model=model)
    return answer, chunks


# ============ Streamlit UI ============
st.set_page_config(page_title="Fordham RAG", page_icon="🎓")
st.title("Fordham University Q&A")
st.caption("Ask questions about Fordham. Powered by RAG (Retrieval Augmented Generation).")

chunks_to_embed, chunk_embeddings = load_chunks_and_embeddings()
st.success(f"Loaded {len(chunks_to_embed):,} chunks.")

question = st.text_input("Your question", placeholder="e.g. How do I apply for financial aid?")
k = st.slider("Number of chunks to retrieve", 1, 10, 5)

if question:
    with st.spinner("Searching and generating..."):
        answer, retrieved_chunks = rag(question, chunks_to_embed, chunk_embeddings, k=k)
    st.subheader("Answer")
    st.write(answer)

    # Show source pages used for this answer
    if retrieved_chunks:
        st.subheader("Sources used")
        seen_sources = set()
        for c in retrieved_chunks:
            page_name = c.get("page_name", "Unknown page")
            source_url = c.get("source_url", "").strip()
            key = (page_name, source_url)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            if source_url:
                st.markdown(f"- **{page_name}**  \n  {source_url}")
            else:
                st.markdown(f"- **{page_name}**")
