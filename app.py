"""
╔══════════════════════════════════════════════════════════════════════╗
║           AI-Powered Study Buddy — Local NotebookLM Clone           ║
║                                                                      ║
║  Architecture:                                                       ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  Document Ingestion → Chunking → TF-IDF Vectorization (ML)  │    ║
║  │       ↓                                                      │    ║
║  │  Query → Cosine Similarity Retrieval (RAG)                   │    ║
║  │       ↓                                                      │    ║
║  │  Context + Prompt → Flan-T5 (DL/NLP Transformer)            │    ║
║  │       ↓                                                      │    ║
║  │  Explain / Summarize / Quiz / Flashcards                     │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import os
import re
import json
import pickle
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict

# ── ML: Scikit-learn (TF-IDF + Cosine Similarity for RAG Retrieval) ──
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── DL/NLP: HuggingFace Transformers — Flan-T5 (seq2seq, T5-family) ──
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ── PDF & Text Extraction ──
import PyPDF2

# ── Sentence splitting for chunking ──
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════
MODEL_NAME   = "google/flan-t5-base"   # ~250 MB, runs on CPU comfortably
CHUNK_SIZE   = 200          # words per chunk
CHUNK_OVERLAP = 40          # word overlap between chunks
TOP_K        = 3            # how many chunks to retrieve for context
MAX_NEW_TOK  = 512          # max tokens the model may generate
CACHE_DIR    = Path(".model_cache")

# ═══════════════════════════════════════════════════════════════════════
#  1. DOCUMENT PROCESSING  (ingestion + chunking)
# ═══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-windows for better context retrieval."""
    sentences = sent_tokenize(text)
    words = []
    sentence_boundaries = []  # track which word index starts each sentence
    for s in sentences:
        sentence_boundaries.append(len(words))
        words.extend(s.split())

    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]  # drop tiny chunks

# ═══════════════════════════════════════════════════════════════════════
#  2. ML RETRIEVER  (TF-IDF + Cosine Similarity)
# ═══════════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    """
    Classical ML retrieval component.
    Converts document chunks + query into TF-IDF vectors,
    then ranks chunks by cosine similarity to the query.
    This is the 'R' (Retrieval) in RAG.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            stop_words="english",
            max_features=10_000,
            sublinear_tf=True     # log-scale TF to dampen high-freq terms
        )
        self.chunk_vectors = None
        self.chunks: List[str] = []

    def fit(self, chunks: List[str]):
        self.chunks = chunks
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.chunk_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices
                if scores[i] > 0.01]

# ═══════════════════════════════════════════════════════════════════════
#  3. DEEP LEARNING NLP MODEL  (Flan-T5 — seq2seq Transformer)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Flan-T5 model (first run ~1 min)…")
def load_model():
    """
    Flan-T5: instruction-tuned T5 by Google.
    Architecture: Encoder-Decoder Transformer (similar to original T5)
    Training: fine-tuned on 1800+ NLP tasks via instruction tuning.
    Perfect for: summarization, Q&A, explanation, generation tasks.
    """
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model     = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR
    )
    model.eval()
    return tokenizer, model

def generate(prompt: str, tokenizer, model,
             max_new_tokens: int = MAX_NEW_TOK) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# ═══════════════════════════════════════════════════════════════════════
#  4. PROMPT TEMPLATES  (task-specific instruction prompts for Flan-T5)
# ═══════════════════════════════════════════════════════════════════════

def build_explain_prompt(context: str, concept: str) -> str:
    return f"""Explain the following concept in simple, easy-to-understand terms for a student.
Use the provided context to make the explanation accurate.

Context:
{context}

Concept to explain: {concept}

Simple explanation:"""

def build_summarize_prompt(context: str) -> str:
    return f"""Summarize the following study notes into clear, concise bullet points.
Capture all key ideas, definitions, and important facts.

Study notes:
{context}

Summary:"""

def build_quiz_prompt(context: str, num_q: int = 3) -> str:
    return f"""Create {num_q} multiple-choice quiz questions based on the text below.
For each question provide:
- The question
- 4 options labeled A, B, C, D
- The correct answer

Text:
{context}

Quiz:"""

def build_flashcard_prompt(context: str, num_cards: int = 4) -> str:
    return f"""Create {num_cards} study flashcards from the text below.
Format each flashcard as:
FRONT: [key term or question]
BACK: [definition or answer]

Text:
{context}

Flashcards:"""

# ═══════════════════════════════════════════════════════════════════════
#  5. SESSION STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "retriever": None,
        "doc_text":  "",
        "chunks":    [],
        "history":   [],   # [(role, content), ...]
        "doc_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_document(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_txt(uploaded_file)

    chunks = chunk_text(text)
    retriever = TFIDFRetriever()
    retriever.fit(chunks)

    st.session_state.doc_text  = text
    st.session_state.chunks    = chunks
    st.session_state.retriever = retriever
    st.session_state.doc_loaded = True
    st.session_state.history   = []
    return len(chunks), len(text.split())

def get_context(query: str) -> str:
    results = st.session_state.retriever.retrieve(query, top_k=TOP_K)
    return "\n\n".join(chunk for chunk, score in results)

# ═══════════════════════════════════════════════════════════════════════
#  6. STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AI Study Buddy",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #3EC6E0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { color: #888; font-size: 1rem; margin-top: -12px; }
    .card {
        background: #1E1E2E; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 12px;
        border-left: 4px solid #6C63FF;
    }
    .user-msg {
        background: #2A2A3E; border-radius: 10px;
        padding: 10px 14px; margin: 6px 0;
        border-left: 3px solid #3EC6E0;
    }
    .bot-msg {
        background: #1A2A1A; border-radius: 10px;
        padding: 10px 14px; margin: 6px 0;
        border-left: 3px solid #6C63FF;
    }
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 99px; font-size: 0.75rem;
        background: #6C63FF22; color: #6C63FF;
        border: 1px solid #6C63FF55; margin-right: 6px;
    }
    .stButton button {
        background: linear-gradient(90deg, #6C63FF, #3EC6E0);
        color: white; border: none; border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    init_session()

    # ── Load model (cached) ──
    tokenizer, model = load_model()

    # ═══════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 📚 Study Buddy")
        st.markdown('<span class="badge">Flan-T5</span>'
                    '<span class="badge">TF-IDF RAG</span>'
                    '<span class="badge">Local</span>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📄 Upload Study Material")
        uploaded = st.file_uploader(
            "Drop your notes here",
            type=["pdf", "txt"],
            help="Upload lecture notes, textbook chapters, or any study material"
        )

        if uploaded:
            with st.spinner("Processing document…"):
                n_chunks, n_words = load_document(uploaded)
            st.success(f"✅ Loaded! {n_words:,} words → {n_chunks} chunks")

        if st.session_state.doc_loaded:
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            top_k = st.slider("Chunks retrieved (RAG Top-K)", 1, 6, TOP_K)
            max_tok = st.slider("Max response tokens", 128, 512, MAX_NEW_TOK, 64)
            st.markdown("---")
            st.markdown("### 📊 Document Stats")
            st.info(f"**Words:** {len(st.session_state.doc_text.split()):,}\n\n"
                    f"**Chunks:** {len(st.session_state.chunks)}")

            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            top_k  = TOP_K
            max_tok = MAX_NEW_TOK

        st.markdown("---")
        st.markdown("""
        <small>
        **Model:** google/flan-t5-base<br>
        **Retrieval:** TF-IDF + Cosine Similarity<br>
        **Generation:** Seq2Seq Transformer<br>
        **Runs:** 100% locally, no API needed
        </small>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN PANEL
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="main-title">📖 AI Study Buddy</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Your local AI tutor — explain, summarize, quiz, flashcards</div>',
                unsafe_allow_html=True)
    st.markdown("")

    if not st.session_state.doc_loaded:
        # ── Welcome screen ──
        col1, col2, col3, col4 = st.columns(4)
        for col, icon, title, desc in [
            (col1, "🧠", "Explain",    "Ask to explain any concept from your notes in simple language"),
            (col2, "📝", "Summarize",  "Get a concise bullet-point summary of your study material"),
            (col3, "❓", "Quiz",       "Auto-generate multiple-choice questions to test yourself"),
            (col4, "🃏", "Flashcards", "Create term/definition flashcards for active recall"),
        ]:
            with col:
                st.markdown(f"""
                <div class="card">
                <h3>{icon} {title}</h3>
                <p style="color:#aaa;font-size:0.85rem">{desc}</p>
                </div>""", unsafe_allow_html=True)

        st.info("👈 Upload a PDF or TXT file in the sidebar to get started!")
        st.markdown("### 🏗️ System Architecture")
        st.code("""
┌─────────────────────────────────────────────────────────┐
│                  AI Study Buddy Pipeline                │
│                                                         │
│  📄 Document                                            │
│      │                                                  │
│      ▼                                                  │
│  [Text Extraction] (PyPDF2 / plain text)                │
│      │                                                  │
│      ▼                                                  │
│  [Chunking] — overlapping word windows (NLTK)           │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────────────┐                               │
│  │  ML LAYER            │  ← TF-IDF Vectorizer          │
│  │  TF-IDF + Cosine Sim │    (sklearn, n-gram 1-2)      │
│  │  (RAG Retrieval)     │                               │
│  └──────────┬───────────┘                               │
│             │ Top-K relevant chunks                     │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │  DL/NLP LAYER        │  ← Flan-T5 (encoder-decoder)  │
│  │  google/flan-t5-base │    Seq2Seq Transformer        │
│  │  (Text Generation)   │    Beam Search decoding       │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  📤 Explain / Summarize / Quiz / Flashcards             │
└─────────────────────────────────────────────────────────┘
        """, language="")
        return

    # ── MODE SELECTOR ──
    st.markdown("### 🎯 Choose a Mode")
    mode_cols = st.columns(4)
    mode = None
    modes = [
        ("🧠 Explain", "explain"),
        ("📝 Summarize", "summarize"),
        ("❓ Quiz", "quiz"),
        ("🃏 Flashcards", "flashcards"),
    ]
    for col, (label, val) in zip(mode_cols, modes):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["mode"] = val

    if "mode" not in st.session_state:
        st.session_state["mode"] = "explain"

    mode = st.session_state["mode"]
    st.markdown(f"**Active mode:** `{mode}`")
    st.markdown("---")

    # ── CHAT HISTORY ──
    for role, content in st.session_state.history:
        if role == "user":
            st.markdown(f'<div class="user-msg">🧑‍🎓 <b>You:</b> {content}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 <b>Study Buddy:</b><br>{content}</div>',
                        unsafe_allow_html=True)

    # ── INPUT ──
    st.markdown("### 💬 Your Query")

    if mode == "explain":
        user_input = st.text_input(
            "What concept would you like explained?",
            placeholder="e.g. 'What is mitosis?' or 'Explain Newton's second law'"
        )
        num_q = 3  # unused
    elif mode == "summarize":
        user_input = st.text_input(
            "Topic to summarize (or leave blank for full summary)",
            placeholder="e.g. 'chapter 3' or leave blank"
        )
        num_q = 3
    elif mode == "quiz":
        user_input = st.text_input(
            "Topic for quiz questions",
            placeholder="e.g. 'photosynthesis' or 'World War II'"
        )
        num_q = st.slider("Number of questions", 2, 6, 3)
    else:  # flashcards
        user_input = st.text_input(
            "Topic for flashcards",
            placeholder="e.g. 'key vocabulary' or 'important dates'"
        )
        num_q = st.slider("Number of flashcards", 2, 8, 4)

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run = st.button("🚀 Generate", use_container_width=True)

    if run and user_input.strip():
        query = user_input.strip()
        context = get_context(query if query else "summary overview")

        if not context:
            st.warning("Could not retrieve relevant context. Try a different query.")
            return

        # ── Build prompt & generate ──
        with st.spinner("🤔 Thinking with Flan-T5…"):
            if mode == "explain":
                prompt = build_explain_prompt(context, query)
                label  = f"Explanation of: *{query}*"
            elif mode == "summarize":
                # for full summary, use a broader context
                if not query or len(query) < 3:
                    context = "\n\n".join(
                        st.session_state.chunks[:min(6, len(st.session_state.chunks))]
                    )
                prompt = build_summarize_prompt(context)
                label  = "Summary"
            elif mode == "quiz":
                prompt = build_quiz_prompt(context, num_q)
                label  = f"Quiz on: *{query}*"
            else:
                prompt = build_flashcard_prompt(context, num_q)
                label  = f"Flashcards for: *{query}*"

            response = generate(prompt, tokenizer, model, max_new_tokens=max_tok)

        # ── Display ──
        st.markdown(f"#### 📤 {label}")
        if mode == "flashcards":
            # parse and display as cards
            cards = re.split(r"FRONT:", response)
            displayed = False
            for card in cards:
                if "BACK:" in card:
                    parts = card.split("BACK:")
                    front = parts[0].strip()
                    back  = parts[1].strip() if len(parts) > 1 else ""
                    st.markdown(f"""
                    <div class="card">
                    🃏 <b>FRONT:</b> {front}<br><br>
                    💡 <b>BACK:</b> {back}
                    </div>""", unsafe_allow_html=True)
                    displayed = True
            if not displayed:
                st.markdown(f'<div class="card">{response}</div>',
                            unsafe_allow_html=True)
        elif mode == "quiz":
            st.markdown(f'<div class="card">{response.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card">{response}</div>',
                        unsafe_allow_html=True)

        # ── Save history ──
        st.session_state.history.append(("user", f"[{mode}] {query}"))
        st.session_state.history.append(("bot",  response))

        # ── Show retrieved chunks (debug / transparency) ──
        with st.expander("🔍 Retrieved context chunks (RAG transparency)"):
            results = st.session_state.retriever.retrieve(query, top_k=top_k)
            for i, (chunk, score) in enumerate(results, 1):
                st.markdown(f"**Chunk {i}** (similarity score: `{score:.4f}`)")
                st.text(textwrap.fill(chunk[:400], width=90) + ("…" if len(chunk) > 400 else ""))
                st.markdown("---")

if __name__ == "__main__":
    main()
