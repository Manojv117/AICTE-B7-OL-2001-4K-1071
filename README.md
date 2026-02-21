# 📚 AI Study Buddy — Local NotebookLM Clone

> A fully local, AI-powered study assistant. No internet, no API keys, no cost.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AI Study Buddy Pipeline                       │
│                                                                     │
│   📄 PDF / TXT                                                      │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────────────────────────────────────────────────┐          │
│   │  TEXT EXTRACTION LAYER                               │          │
│   │  PyPDF2 (PDF) / plain decode (TXT)                  │          │
│   └────────────────────────┬─────────────────────────────┘          │
│                            │                                        │
│        ▼                                                            │
│   ┌──────────────────────────────────────────────────────┐          │
│   │  CHUNKING LAYER (NLTK sentence tokenizer)           │          │
│   │  Overlapping word-windows (200 words, 40 overlap)   │          │
│   └────────────────────────┬─────────────────────────────┘          │
│                            │                                        │
│        ▼                                                            │
│   ┌──────────────────────────────────────────────────────┐          │
│   │  ML RETRIEVAL LAYER  ← CLASSICAL MACHINE LEARNING   │          │
│   │  Algorithm: TF-IDF Vectorizer (scikit-learn)        │          │
│   │  • n-grams: 1–2, sublinear_tf=True                 │          │
│   │  • 10,000 features, English stop words removed     │          │
│   │  Similarity: Cosine Similarity                      │          │
│   │  → Returns Top-K most relevant chunks for query    │          │
│   └────────────────────────┬─────────────────────────────┘          │
│                            │  Retrieved context chunks              │
│        ▼                                                            │
│   ┌──────────────────────────────────────────────────────┐          │
│   │  PROMPT ENGINEERING LAYER                            │          │
│   │  Task-specific instruction templates:               │          │
│   │  • Explain  • Summarize  • Quiz  • Flashcards       │          │
│   └────────────────────────┬─────────────────────────────┘          │
│                            │                                        │
│        ▼                                                            │
│   ┌──────────────────────────────────────────────────────┐          │
│   │  DEEP LEARNING / NLP GENERATION LAYER               │          │
│   │  Model: google/flan-t5-base (~250MB)                │          │
│   │  Architecture: Encoder-Decoder Transformer (T5)     │          │
│   │  Training: Instruction-tuned on 1800+ NLP tasks     │          │
│   │  Decoding: Beam Search (4 beams), no-repeat n-gram  │          │
│   │  Runs: CPU only, no GPU needed                      │          │
│   └────────────────────────┬─────────────────────────────┘          │
│                            │                                        │
│        ▼                                                            │
│   📤 Generated Output → Streamlit UI                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 ML & DL Components Explained

### Classical ML — TF-IDF Retriever
| Property | Detail |
|---|---|
| Algorithm | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Purpose | Convert text chunks and queries into numeric vectors |
| Similarity | Cosine similarity to rank chunks by relevance |
| Role in system | RAG retrieval — find the most relevant part of your notes |
| Library | `scikit-learn` |
| Why this? | Fast, interpretable, no GPU, works well on domain-specific text |

**How TF-IDF works:**
- TF (Term Frequency): how often a word appears in a chunk
- IDF (Inverse Document Frequency): penalizes words common across all chunks
- Together they highlight unique, meaningful terms per chunk
- Cosine similarity measures the angle between two TF-IDF vectors

### Deep Learning NLP — Flan-T5 (Transformer)
| Property | Detail |
|---|---|
| Model | `google/flan-t5-base` (~250 MB) |
| Architecture | Encoder-Decoder Transformer (T5 family) |
| Training | Instruction-tuned by Google on 1800+ diverse NLP tasks |
| Input | Text prompt with context + instruction |
| Output | Generated text (explanation, summary, quiz, flashcards) |
| Library | HuggingFace `transformers` |
| Decoding | Beam search (4 beams), temperature=0.7, no repeat n-gram |
| Hardware | CPU only — no GPU required |

**Why Flan-T5 over GPT-2 or BERT?**
- GPT-2: Causal LM, not instruction-following by default
- BERT: Encoder-only, great for classification/extraction but poor generation
- Flan-T5: Encoder-Decoder + instruction-tuning = ideal for task-following (explain, summarize, quiz generation)

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. First run
- The app downloads Flan-T5 (~250 MB) on first launch and caches it locally.
- After that, it starts in seconds.

### 4. Upload your notes
- Use `sample_notes.txt` (Biology notes) included to test.
- Or upload any `.pdf` or `.txt` study material.

---

## 🎯 Features

| Feature | What it does |
|---|---|
| 🧠 **Explain** | Explain any concept from your notes in simple terms |
| 📝 **Summarize** | Get bullet-point summaries of any topic or full notes |
| ❓ **Quiz** | Auto-generate multiple-choice questions to test yourself |
| 🃏 **Flashcards** | Create FRONT/BACK flashcards for active recall |
| 🔍 **RAG Transparency** | See exactly which chunks were retrieved and their scores |

---

## 🔧 Configuration (in app.py)

```python
MODEL_NAME    = "google/flan-t5-base"  # swap for flan-t5-large for better quality
CHUNK_SIZE    = 200    # words per chunk
CHUNK_OVERLAP = 40     # overlap between chunks
TOP_K         = 3      # retrieved chunks per query
MAX_NEW_TOK   = 512    # max generation length
```

### Upgrade model for better quality:
```python
MODEL_NAME = "google/flan-t5-large"   # ~770 MB, noticeably better
MODEL_NAME = "google/flan-t5-xl"      # ~3 GB, excellent quality (needs more RAM)
```

---

## 📦 System Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| Storage | 1 GB | 2 GB |
| GPU | Not required | Optional (speeds up generation) |
| Python | 3.9+ | 3.11+ |

---

## 📂 Project Structure
```
study_buddy/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── sample_notes.txt    ← Biology notes for testing
├── README.md           ← This file
└── .model_cache/       ← Auto-created, stores downloaded model
```
