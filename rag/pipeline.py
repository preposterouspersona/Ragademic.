import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import Retriever
from .llm import GroqLLM

load_dotenv()


class RAGPipeline:

    def __init__(
            self,
            vector_db_path: str = "./data/vector_db",
            chunk_size: int = 1500,
            chunk_overlap: int = 300,
            top_k: int = 7,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(path=vector_db_path)
        self.retriever = Retriever(self.vector_store, self.embedding_manager)
        self.llm = GroqLLM()

    def index_pdfs(self, pdf_directory: str):
        pdf_path = Path(pdf_directory)
        pdf_list = [p for p in pdf_path.glob("**/*.pdf") if "-checkpoint" not in p.name]
        print(f"Found {len(pdf_list)} PDFs")

        all_docs = []
        for pdf in pdf_list:
            print(f"Loading {pdf.name}...")
            try:
                docs = PyPDFLoader(str(pdf)).load()
                for doc in docs:
                    doc.metadata["source"] = pdf.name
                    doc.metadata["file_type"] = "pdf"
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf.name}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)
        print(f"Split into {len(chunks)} chunks")

        text_content = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.create_embeddings(text_content)
        self.vector_store.push_docs(chunks, embeddings)
        return len(chunks)

    def index_uploaded_file(self, file_path: str):
        """Index a single uploaded PDF file (used by Streamlit)."""
        try:
            docs = PyPDFLoader(file_path).load()
            for doc in docs:
                doc.metadata["source"] = Path(file_path).name
                doc.metadata["file_type"] = "pdf"

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            text_content = [doc.page_content for doc in chunks]
            embeddings = self.embedding_manager.create_embeddings(text_content)
            self.vector_store.push_docs(chunks, embeddings)
            return len(chunks)
        except Exception as e:
            print(f"Error indexing file: {e}")
            return 0

    def query(self, question: str) -> dict:
        """Retrieve context and generate answer."""
        results = self.retriever.retrieve(question, top_k=self.top_k)

        if not results:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": []
            }

        context = "\n\n".join([r["content"] for r in results])
        answer = self.llm.generate(question, context)

        sources = [
            {
                "source": r["metadata"].get("source", "unknown"),
                "page": r["metadata"].get("page", "?"),
                "score": round(r["similarity_score"], 3)
            }
            for r in results
        ]

        return {"answer": answer, "sources": sources}

    def reset(self):
        self.vector_store.reset_collection()
