

# Ragademic

> A full-stack RAG (Retrieval-Augmented Generation) pipeline that lets you query your own documents using semantic search and large language models. Upload a PDF, ask a question, get a precise answer grounded in your content — with source citations.



<!-- Replace with actual screenshot/gif -->

---

https://github.com/user-attachments/assets/c85e0743-32ab-43d2-9ce2-8ccfc9168950



## Features

- **Upload any PDF** — textbooks, lecture notes, research papers, slides
- **Ask in plain English** — no keywords, no search syntax, just questions
- **No hallucinations** — the model is strictly prompted to answer only from retrieved context. If the answer isn't in your documents, it says so instead of making something up.
- **Answers grounded in your content** — the model only uses what's in your documents
- **Source citations** — every answer links back to the exact page it came from
- **Multi-document support** — index multiple PDFs and query across all of them at once
- **Session chat history** — full conversation context preserved within a session
- **One-click reset** — wipe the vector store and start fresh instantly

---

## What's actually happening under the hood

Most "RAG apps" are thin wrappers around LangChain's `RetrievalQA`. Ragademic is built from scratch — every layer of the pipeline is implemented explicitly so the system is fully inspectable and extensible.

### 1. Document Ingestion
PDFs are loaded using `PyPDFLoader` and split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200). Overlapping ensures that concepts straddling chunk boundaries aren't lost. Each chunk retains its source filename, page number, and character length as metadata.

### 2. Semantic Embedding
Chunks are encoded using `BAAI/bge-base-en-v1.5`, a retrieval-optimised sentence transformer that outperforms general-purpose models like `all-MiniLM-L6-v2` on technical and scientific content. Embeddings are L2-normalised before storage, enabling accurate cosine similarity comparisons. The model runs entirely locally — no API calls, no cost.

### 3. Vector Storage
Embeddings are persisted in **ChromaDB** with `hnsw:space: cosine` set explicitly on the collection. Each document chunk is stored with its embedding, raw text, and metadata as a single atomic record. ChromaDB's HNSW index enables sub-linear approximate nearest neighbour search at query time.

### 4. Retrieval
At query time, the user's question is encoded using the same BGE model with a retrieval-specific prefix:
```
"Represent this sentence for searching relevant passages: {query}"
```
This prefix is part of BGE's training protocol and measurably improves retrieval precision for technical queries. The query vector is compared against all stored chunk vectors using cosine similarity. The top-k most similar chunks are returned, deduplicated by content fingerprint, and ranked by similarity score.

### 5. Generation
Retrieved chunks are assembled into a context window and passed to **Llama 3.3 70B** via the **Groq API** — chosen for its best-in-class inference speed (typically under 1 second). A structured prompt grounds the model strictly in the retrieved context, minimising hallucinations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document Loading | LangChain `PyPDFLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers` |
| Vector Store | ChromaDB (persistent, cosine space) |
| LLM | Llama 3.3 70B via Groq API |
| LLM Framework | LangChain Core + LangChain Groq |
| Frontend | Streamlit |

---

## Project Structure
```
ragademic/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .gitignore
├── README.md
└── rag/
    ├── __init__.py
    ├── embeddings.py       # EmbeddingManager — loads BGE, encodes text
    ├── vector_store.py     # VectorStore — ChromaDB wrapper
    ├── retriever.py        # Retriever — semantic search + deduplication
    ├── llm.py              # GroqLLM — prompt formatting + generation
    └── pipeline.py         # RAGPipeline — orchestrates all components
```

---

## Screenshots

![Upload](assets/upload.png)
<!-- Add screenshot of PDF upload flow -->

![Chat](assets/chat.png)
<!-- Add screenshot of chat with sources -->

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/preposterouspersona/Ragademic.git
cd Ragademic
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

**5. Run**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. Upload one or more PDFs using the sidebar
2. Click **Index PDFs** — chunks are embedded and stored in ChromaDB
3. Ask questions in the chat
4. Answers are grounded in your documents with page-level source citations

---

## Future Roadmap and Plans.

### Deployment
- Host on **Hugging Face Spaces** (Streamlit SDK) for a free public demo
- Swap ChromaDB `PersistentClient` for `EphemeralClient` for stateless cloud deployment

### Scale
- Replace ChromaDB with a hosted vector DB (**Pinecone / Qdrant Cloud**) for multi-user, persistent storage
- Add **rate limiting** via Redis to prevent API quota exhaustion


### Features
- **Conversational memory** — multi-turn context so follow-up questions work naturally
- **Multi-modal** — extract and query content from images and tables within PDFs
- **Export** — download Q&A sessions as formatted notes

---

## Notes

- The embedding model (`bge-base-en-v1.5`) downloads on first run (~400MB, cached after)
- ChromaDB persists to `./data/vector_db` — re-indexing is only needed when you change documents or the embedding model
- To reset the vector store, click the **Reset** button in the sidebar
