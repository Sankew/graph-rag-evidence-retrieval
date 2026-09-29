"""Behavioral checks for evidence-backed, permission-aware graph retrieval."""

import unittest

from rag_graph.access import ACLStore, can_read
from rag_graph.domain import ACL, Chunk, Principal
from rag_graph.graph import EvidenceGraph, canonicalize_entity, combine_retrieval


def make_chunk(chunk_id: str, text: str, *, tenant: str = "acme") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        text=text,
        acl=ACL(tenant, frozenset({"reader"}), frozenset()),
        version=1,
        embedding=(1.0, 0.0),
    )


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = Principal("ana", "acme", frozenset({"reader"}))
        self.a = make_chunk("a", "Ada founded Northstar.")
        self.b = make_chunk("b", "Northstar acquired Bluebird.")
        self.c = make_chunk("c", "Bluebird is based in Perth.")
        self.chunks = {chunk.chunk_id: chunk for chunk in (self.a, self.b, self.c)}
        self.graph = EvidenceGraph()
        self.graph.add_relation("Ada", "founded", "Northstar", "a")
        self.graph.add_relation("Northstar", "acquired", "Bluebird", "b")
        self.graph.add_relation("Bluebird", "located in", "Perth", "c")

    def test_canonicalization_and_two_hop_provenance(self) -> None:
        self.assertEqual(canonicalize_entity("  NORTHSTAR—Inc.  "), "northstar inc")
        hits = self.graph.expand(
            [" ADA "], self.chunks, self.principal,
            lambda principal, chunk: can_read(principal, chunk.acl), max_hops=2,
        )
        by_id = {hit.chunk.chunk_id: hit for hit in hits}
        self.assertEqual(set(by_id), {"a", "b"})
        self.assertEqual(by_id["b"].hop, 2)
        self.assertEqual([edge.chunk_id for edge in by_id["b"].evidence], ["a", "b"])

    def test_unauthorized_edge_cannot_bridge_entities(self) -> None:
        private = make_chunk("secret", "Northstar acquired Bluebird.", tenant="other")
        graph = EvidenceGraph()
        graph.add_relation("Ada", "founded", "Northstar", "a")
        graph.add_relation("Northstar", "acquired", "Bluebird", "secret")
        graph.add_relation("Bluebird", "located in", "Perth", "c")
        chunks = {"a": self.a, "secret": private, "c": self.c}
        hits = graph.expand(
            ["Ada"], chunks, self.principal,
            lambda principal, chunk: can_read(principal, chunk.acl), max_hops=2,
        )
        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["a"])

    def test_revocation_is_checked_at_query_time(self) -> None:
        store = ACLStore()
        for chunk in self.chunks.values():
            store.update(chunk.document_id, chunk.acl)

        def authorized(principal: Principal, chunk: Chunk) -> bool:
            current = store.get(chunk.document_id)
            return current is not None and can_read(principal, chunk.acl) and can_read(principal, current)

        before = self.graph.expand(["Ada"], self.chunks, self.principal, authorized)
        self.assertIn("b", {hit.chunk.chunk_id for hit in before})
        store.delete("a")
        after = self.graph.expand(["Ada"], self.chunks, self.principal, authorized)
        self.assertEqual(after, [])

    def test_merge_respects_budget_and_reranker_cannot_inject_chunks(self) -> None:
        rogue = make_chunk("rogue", "Unindexed text")
        result = combine_retrieval(
            "Where is the acquired company?", self.principal, [self.a],
            self.graph, self.chunks, ["Northstar"],
            lambda principal, chunk: can_read(principal, chunk.acl),
            budget=2,
            rerank=lambda _query, candidates: [rogue, self.b, self.b, *candidates],
        )
        self.assertEqual([chunk.chunk_id for chunk in result], ["b", "a"])

    def test_invalid_budgets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.graph.expand(["Ada"], self.chunks, self.principal,
                              lambda _principal, _chunk: True, max_hops=3)
        with self.assertRaises(ValueError):
            combine_retrieval("q", self.principal, [], self.graph, self.chunks,
                              [], lambda _principal, _chunk: True, budget=0)

    def test_remove_chunk_invalidates_edges_and_revision(self) -> None:
        before = self.graph.revision
        self.graph.remove_chunk("b")
        self.assertGreater(self.graph.revision, before)
        hits = self.graph.expand(["Ada"], self.chunks, self.principal, lambda _principal, _chunk: True)
        self.assertNotIn("b", {hit.chunk.chunk_id for hit in hits})

    def test_graph_source_does_not_get_reserved_slot_without_reranker(self) -> None:
        result = combine_retrieval(
            "What did Northstar acquire?", self.principal, [self.a, self.c],
            self.graph, self.chunks, ["Northstar"],
            lambda _principal, _chunk: True, budget=2,
        )
        self.assertEqual([chunk.chunk_id for chunk in result], ["a", "c"])

    def test_reranker_can_select_graph_source(self) -> None:
        result = combine_retrieval(
            "What did Northstar acquire?", self.principal, [self.a, self.c],
            self.graph, self.chunks, ["Northstar"],
            lambda _principal, _chunk: True, budget=2,
            rerank=lambda _query, candidates: sorted(candidates, key=lambda chunk: chunk.chunk_id),
        )
        self.assertEqual([chunk.chunk_id for chunk in result], ["a", "b"])

    def test_reranker_can_omit_graph_source(self) -> None:
        result = combine_retrieval(
            "What did Northstar acquire?", self.principal, [self.a, self.c],
            self.graph, self.chunks, ["Northstar"],
            lambda _principal, _chunk: True, budget=2,
            rerank=lambda _query, candidates: [
                chunk for chunk in candidates if chunk.chunk_id in {"a", "c"}
            ],
        )
        self.assertEqual([chunk.chunk_id for chunk in result], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
