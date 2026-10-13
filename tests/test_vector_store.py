from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.services.vector_store import ChromaVectorStore, VectorRecord
from tests.fakes import FakeEmbedder


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma")


def add_texts(
    store: ChromaVectorStore,
    embedder: FakeEmbedder,
    document_id: str,
    texts: list[str],
    page: int | None = 1,
) -> list[str]:
    ids = [f"{document_id}-{index}" for index in range(len(texts))]
    embeddings = embedder.embed_documents(texts)
    store.add(
        [
            VectorRecord(chunk_id, embedding, document_id, index, page, page)
            for index, (chunk_id, embedding) in enumerate(zip(ids, embeddings, strict=True))
        ]
    )
    return ids


def test_search_ranks_by_similarity(store: ChromaVectorStore, embedder: FakeEmbedder) -> None:
    add_texts(
        store,
        embedder,
        "doc",
        [
            "Q3 revenue grew 12 percent",
            "Employee handbook vacation policy",
            "Revenue drivers in Q3 were enterprise deals",
        ],
    )

    hits = store.search(embedder.embed_query("Q3 revenue"), top_k=3, document_ids=["doc"])

    assert [hit.id for hit in hits] == ["doc-0", "doc-2", "doc-1"]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_identical_text_scores_one(store: ChromaVectorStore, embedder: FakeEmbedder) -> None:
    add_texts(store, embedder, "doc", ["Employee handbook vacation policy", "Q3 revenue grew"])

    hits = store.search(
        embedder.embed_query("Employee handbook vacation policy"), top_k=1, document_ids=["doc"]
    )

    assert hits[0].id == "doc-0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_returns_at_most_top_k(store: ChromaVectorStore, embedder: FakeEmbedder) -> None:
    add_texts(store, embedder, "doc", [f"revenue report part {n}" for n in range(10)])

    hits = store.search(embedder.embed_query("revenue report"), top_k=3, document_ids=["doc"])

    assert len(hits) == 3


def test_search_is_restricted_to_requested_documents(
    store: ChromaVectorStore, embedder: FakeEmbedder
) -> None:
    add_texts(store, embedder, "doc-a", ["Q3 revenue grew", "Q3 revenue fell"])
    add_texts(store, embedder, "doc-b", ["Q3 revenue was flat", "Q3 revenue forecast"])
    query = embedder.embed_query("Q3 revenue")

    only_a = store.search(query, top_k=10, document_ids=["doc-a"])
    both = store.search(query, top_k=10, document_ids=["doc-a", "doc-b"])

    assert {hit.document_id for hit in only_a} == {"doc-a"}
    assert {hit.document_id for hit in both} == {"doc-a", "doc-b"}


def test_delete_document_removes_only_its_vectors(
    store: ChromaVectorStore, embedder: FakeEmbedder
) -> None:
    add_texts(store, embedder, "doc-a", ["alpha one", "alpha two"])
    add_texts(store, embedder, "doc-b", ["beta one"])

    store.delete_document("doc-a")

    assert store.count("doc-a") == 0
    assert store.count("doc-b") == 1
    assert store.count() == 1


def test_records_without_pages_are_stored(store: ChromaVectorStore, embedder: FakeEmbedder) -> None:
    add_texts(store, embedder, "notes", ["plain text notes"], page=None)

    hits = store.search(embedder.embed_query("plain text"), top_k=5, document_ids=["notes"])

    assert [hit.id for hit in hits] == ["notes-0"]


def test_empty_document_filter_searches_nothing(
    store: ChromaVectorStore, embedder: FakeEmbedder
) -> None:
    add_texts(store, embedder, "doc", ["Q3 revenue grew"])

    assert store.search(embedder.embed_query("Q3 revenue"), top_k=5, document_ids=[]) == []


def test_search_on_empty_collection_returns_nothing(
    store: ChromaVectorStore, embedder: FakeEmbedder
) -> None:
    assert store.search(embedder.embed_query("anything"), top_k=5, document_ids=["doc"]) == []


def test_concurrent_adds_deletes_and_searches_from_threads(
    store: ChromaVectorStore, embedder: FakeEmbedder
) -> None:
    query = embedder.embed_query("revenue")

    def write(worker: int) -> None:
        for doc in range(10):
            document_id = f"w{worker}-d{doc}"
            add_texts(store, embedder, document_id, [f"revenue part {n}" for n in range(5)])
            if doc % 2 == 0:
                store.delete_document(document_id)

    def read(_: int) -> None:
        for _ in range(50):
            store.search(query, top_k=5, document_ids=[f"w0-d{doc}" for doc in range(10)])

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(write, w) for w in range(4)] + [
            pool.submit(read, r) for r in range(2)
        ]
        for future in futures:
            future.result()

    assert store.count() == 4 * 5 * 5
    assert all(
        store.count(f"w{w}-d{d}") == (0 if d % 2 == 0 else 5) for w in range(4) for d in range(10)
    )
