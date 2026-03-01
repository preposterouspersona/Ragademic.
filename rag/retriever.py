from typing import List, Dict, Any
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.embedding_manager.create_embeddings([prefixed_query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    similarity_score = 1 - (distance / 2)
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

            # deduplicate by content fingerprint
            seen, deduped = set(), []
            for doc in retrieved_docs:
                key = doc["content"][:100]
                if key not in seen:
                    seen.add(key)
                    deduped.append(doc)

            return deduped

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
