import streamlit as st
import tempfile
import os
from rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="Ragademic",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    </style>
    <h1 style="font-family: 'Inter', sans-serif; font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px; margin-bottom: 4px;">
    Ragademic
    </h1>
    <p style="font-family: 'Inter', sans-serif; font-size: 0.95rem; 
    color: #9ca3af; font-weight: 400; margin-top: 0px; letter-spacing: 0.3px;">
    Last minute cramming? I got you.
    </p>
    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0 24px 0;">
""", unsafe_allow_html=True)
# ── Initialize pipeline in session state ─────────────────────────────────────
if "pipeline" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ── Sidebar — PDF upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

        if new_files:
            if st.button("📥 Index PDFs", type="primary"):
                for uploaded_file in new_files:
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        chunks = st.session_state.pipeline.index_uploaded_file(tmp_path)
                        os.unlink(tmp_path)
                        st.session_state.indexed_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} — {chunks} chunks indexed")
        else:
            st.info("All uploaded files are already indexed.")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Indexed files")
        for f in st.session_state.indexed_files:
            st.markdown(f"- {f}")

        if st.button("🗑️ Reset DB", type="secondary"):
            st.session_state.pipeline.reset()
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.success("Database reset!")

# ── Chat interface ────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for s in message["sources"]:
                    st.markdown(f"- **{s['source']}** — page {s['page']} (score: {s['score']})")

if prompt := st.chat_input("Ask a question about your PDFs..."):
    if not st.session_state.indexed_files:
        st.warning("Please upload and index a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.pipeline.query(prompt)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📚 Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source']}** — page {s['page']} (score: {s['score']})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
