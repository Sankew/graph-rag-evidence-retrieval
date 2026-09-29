"""A small permission-aware vector and graph question-answering service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .access import ACLStore, can_read, secure_retrieve
from .baseline import Embedder, Generator, MemoryIndex
from .domain import Answer, Chunk, Principal, ScoredChunk
from .graph import EvidenceGraph, combine_retrieval


@dataclass(frozen=True)
class ServiceResult:
    answer: Answer
    retrieved: tuple[ScoredChunk, ...]
    latency_ms: float


class GraphRAGService:
    def __init__(
        self,
        *,
        index: MemoryIndex,
        acl_store: ACLStore,
        embedder: Embedder,
        generator: Generator,
        top_k: int = 5,
        candidate_pool: int = 25,
        graph: EvidenceGraph | None = None,
        question_entities: Callable[[str], Sequence[str]] | None = None,
        reranker: Callable[[str, Sequence[Chunk]], Sequence[Chunk]] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if candidate_pool < top_k:
            raise ValueError("candidate_pool must be at least top_k")
        if graph is not None and question_entities is None:
            raise ValueError("graph retrieval requires a question entity linker")
        self.index, self.acl_store, self.embedder = index, acl_store, embedder
        self.generator, self.top_k, self.candidate_pool = generator, top_k, candidate_pool
        self.graph, self.question_entities, self.reranker = graph, question_entities, reranker

    def answer(self, principal: Principal, question: str) -> ServiceResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        start = time.perf_counter()
        vector = self.embedder.embed(question)
        ranked = self.index.query(vector, limit=max(self.top_k, len(self.index.all_chunks())))
        # The vector arm reranks 25 authorized candidates. The graph arm starts
        # from five vector candidates and may add up to 20 graph candidates.
        vector_limit = self.candidate_pool if self.graph is None and self.reranker else self.top_k
        authorized = secure_retrieve(
            principal, (hit.chunk for hit in ranked), self.acl_store, limit=vector_limit
        )

        def can_read_now(user: Principal, chunk: Chunk) -> bool:
            current = self.acl_store.get(chunk.document_id)
            return current is not None and can_read(user, chunk.acl) and can_read(user, current)

        if self.graph is not None:
            assert self.question_entities is not None
            authorized = tuple(combine_retrieval(
                question, principal, authorized, self.graph,
                {chunk.chunk_id: chunk for chunk in self.index.all_chunks()},
                self.question_entities(question), can_read_now,
                budget=self.top_k,
                max_graph_chunks=self.candidate_pool - self.top_k,
                rerank=self.reranker,
            ))
        elif self.reranker is not None:
            candidates = {chunk.chunk_id: chunk for chunk in authorized}
            proposed = self.reranker(question, authorized)
            authorized = tuple(candidates[chunk.chunk_id] for chunk in proposed if chunk.chunk_id in candidates)[:self.top_k]

        score_by_id = {hit.chunk.chunk_id: hit.score for hit in ranked}
        evidence = tuple(ScoredChunk(chunk, score_by_id[chunk.chunk_id]) for chunk in authorized)
        answer = self.generator.generate(question, list(evidence))
        allowed = {(chunk.document_id, chunk.chunk_id, chunk.version) for chunk in authorized}
        if any((cite.document_id, cite.chunk_id, cite.version) not in allowed for cite in answer.citations):
            raise ValueError("generator cited a chunk outside the authorized context")
        return ServiceResult(answer, evidence, (time.perf_counter() - start) * 1000)
