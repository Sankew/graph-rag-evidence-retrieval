from rag_graph.domain import ACL, Chunk
from rag_graph.graph import EvidenceGraph
from rag_graph.graph_extraction import (
    ExtractedRelation,
    add_cooccurrence_evidence,
    add_extracted_relations,
    candidate_entities,
    question_linker,
)


def test_question_linker_uses_explicit_titles_without_gold_answers():
    assert question_linker("Where was the author of The Atlas born?", ["The Atlas", "Other Work"])[0] == "The Atlas"
    assert "Other Work" not in question_linker("Where was the author of The Atlas born?", ["The Atlas", "Other Work"])


def test_cooccurrence_graph_keeps_source_chunk():
    acl = ACL("benchmark", frozenset({"reader"}), frozenset())
    chunk = Chunk("c1", "benchmark:doc", "Ada Lovelace visited Perth.", acl, 1, (1.0,))
    graph = EvidenceGraph()
    assert add_cooccurrence_evidence(graph, chunk, "Ada Lovelace") >= 1
    hits = graph.expand(["Perth"], {"c1": chunk}, object(), lambda principal, source: True)
    assert hits[0].chunk.chunk_id == "c1"


def test_structured_relations_use_fixed_schema():
    acl = ACL("benchmark", frozenset({"reader"}), frozenset())
    chunk = Chunk("c1", "benchmark:doc", "Ada was born in London.", acl, 1, (1.0,))
    graph = EvidenceGraph()
    add_extracted_relations(graph, chunk, [ExtractedRelation("Ada", "born_in", "London")])
    assert graph.expand(["Ada"], {"c1": chunk}, object(), lambda principal, source: True)
