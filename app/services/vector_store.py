from dataclasses import dataclass
from pathlib import Path

import chromadb

COLLECTION_NAME = "chunks"


@dataclass(frozen=True)
class VectorRecord:
    id: str
    embedding: list[float]
    document_id: str
    chunk_index: int
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class SearchHit:
    id: str
    document_id: str
    score: float


class ChromaVectorStore:
    def __init__(self, path: Path) -> None:
        self._client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        # embedding_function=None: vectors always come from our Embedder, so Chroma must not
        # download or apply its own default model.
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def add(self, records: list[VectorRecord]) -> None:
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            self._collection.add(
                ids=[record.id for record in batch],
                embeddings=[record.embedding for record in batch],
                metadatas=[_metadata(record) for record in batch],
            )

    def search(
        self, embedding: list[float], top_k: int, document_ids: list[str]
    ) -> list[SearchHit]:
        """Most similar chunks among `document_ids`, best first. An empty list searches nothing."""
        if not document_ids:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"document_id": {"$in": document_ids}},
            include=["distances", "metadatas"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0] if result["distances"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        # Cosine distance = 1 - cosine similarity, so this recovers the similarity.
        return [
            SearchHit(id=chunk_id, document_id=str(metadata["document_id"]), score=1.0 - distance)
            for chunk_id, distance, metadata in zip(ids, distances, metadatas, strict=True)
        ]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"document_id": document_id}, include=[])["ids"])


def _metadata(record: VectorRecord) -> dict[str, str | int]:
    # Chroma metadata values cannot be None, so missing pages (TXT/MD) are omitted.
    metadata: dict[str, str | int] = {
        "document_id": record.document_id,
        "chunk_index": record.chunk_index,
    }
    if record.page_start is not None:
        metadata["page_start"] = record.page_start
    if record.page_end is not None:
        metadata["page_end"] = record.page_end
    return metadata
