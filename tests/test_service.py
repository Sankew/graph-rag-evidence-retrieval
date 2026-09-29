from rag_graph.access import ACLStore
from rag_graph.baseline import ExtractiveDemoGenerator, HashingEmbedder, MemoryIndex, ingest_document
from rag_graph.domain import ACL, Chunk, Principal
from rag_graph.graph import EvidenceGraph
from rag_graph.service import GraphRAGService


def test_graph_service_expands_evidence_and_obeys_revocation():
    embedder = HashingEmbedder()
    acl = ACL("acme", frozenset({"reader"}), frozenset())
    index, store, graph = MemoryIndex(), ACLStore(), EvidenceGraph()
    first = ingest_document("founder", "Ada founded Northstar.", acl, embedder)[0]
    second = ingest_document("acquisition", "Northstar acquired Bluebird.", acl, embedder)[0]
    index.upsert((first, second))
    store.update(first.document_id, acl)
    store.update(second.document_id, acl)
    graph.add_relation("Ada", "founded", "Northstar", first.chunk_id)
    graph.add_relation("Northstar", "acquired", "Bluebird", second.chunk_id)
    service = GraphRAGService(
        index=index, acl_store=store, embedder=embedder,
        generator=ExtractiveDemoGenerator(), top_k=2,
        graph=graph, question_entities=lambda _: ("Ada",),
    )
    user = Principal("ana", "acme", frozenset({"reader"}))
    before = service.answer(user, "What did Ada found?")
    assert {hit.chunk.chunk_id for hit in before.retrieved} == {first.chunk_id, second.chunk_id}
    store.delete(first.document_id)
    after = service.answer(user, "What did Ada found?")
    assert first.chunk_id not in {hit.chunk.chunk_id for hit in after.retrieved}
    assert all(citation.chunk_id != first.chunk_id for citation in after.answer.citations)


def test_vector_and_graph_arms_rerank_the_same_candidate_ceiling():
    class FixedEmbedder:
        def embed(self, _text):
            return (1.0, 0.0)

    acl = ACL("acme", frozenset({"reader"}), frozenset())
    chunks = [Chunk(name, name, f"{name} source text.", acl, 1, (1.0, 0.0)) for name in "abc"]
    index, store, graph = MemoryIndex(), ACLStore(), EvidenceGraph()
    index.upsert(chunks)
    for chunk in chunks:
        store.update(chunk.document_id, acl)
    graph.add_relation("Target", "related_to", "Other", "c")
    seen = []

    def rerank(_question, candidates):
        seen.append(tuple(chunk.chunk_id for chunk in candidates))
        return sorted(candidates, key=lambda chunk: chunk.chunk_id, reverse=True)

    common = dict(index=index, acl_store=store, embedder=FixedEmbedder(),
                  generator=ExtractiveDemoGenerator(), top_k=1, candidate_pool=2,
                  reranker=rerank)
    user = Principal("ana", "acme", frozenset({"reader"}))
    vector = GraphRAGService(**common).answer(user, "Target")
    expanded = GraphRAGService(**common, graph=graph, question_entities=lambda _: ("Target",)).answer(user, "Target")
    assert seen == [("a", "b"), ("a", "c")]
    assert [hit.chunk.chunk_id for hit in vector.retrieved] == ["b"]
    assert [hit.chunk.chunk_id for hit in expanded.retrieved] == ["c"]
