import os
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import List, Any


class VectorStore:

    def __init__(self, name: str = "Vector_Database", path: str = "./data/vector_db"):
        self.client = None
        self.collection_name = name
        self.path = path
        self.collection = None
        self._create_collection()

    def _create_collection(self):
        try:
            os.makedirs(self.path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.path,
                settings=Settings()
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF sources for RAG",
                    "hnsw:space": "cosine"
                }
            )
            print(f"Collection '{self.collection_name}' ready — {self.collection.count()} docs.")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def push_docs(self, docs: List[Any], embeddings: np.ndarray):
        ids, docs_text, embeddings_list, metadatas = [], [], [], []

        for i, doc in enumerate(docs):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadata = dict(doc.metadata)
            metadata["index"] = i
            metadata["length"] = len(doc.page_content)
            metadatas.append(metadata)
            docs_text.append(doc.page_content)
            embeddings_list.append(embeddings[i].tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=docs_text,
                embeddings=embeddings_list
            )
            print(f"DB updated — {len(docs)} chunks added.")
        except Exception as e:
            print(f"Error updating DB: {e}")

    def reset_collection(self):
        self.client.delete_collection(self.collection_name)
        self._create_collection()
        print("Collection wiped and recreated.")

    def count(self):
        return self.collection.count()
