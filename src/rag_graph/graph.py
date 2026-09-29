"""Evidence-backed graph expansion for multi-hop retrieval.

The graph is deliberately an in-memory index. Persistence and entity extraction are
adapter concerns; every relation here retains the chunk that supports it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Iterable, Mapping, Sequence, TypeVar


ChunkT = TypeVar("ChunkT")
PrincipalT = TypeVar("PrincipalT")


def canonicalize_entity(value: str) -> str:
    """Normalize surface forms without merging distinct named entities."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    subject: str
    predicate: str
    object: str
    chunk_id: str


@dataclass(frozen=True, slots=True)
class GraphHit:
    chunk: object
    hop: int
    evidence: tuple[EvidenceRelation, ...]


class EvidenceGraph:
    """Entity mentions and relations whose edges point to source chunks."""

    def __init__(self) -> None:
        self._mentions: dict[str, set[str]] = defaultdict(set)
        self._relations: list[EvidenceRelation] = []
        self._adjacency: dict[str, list[EvidenceRelation]] = defaultdict(list)
        self.revision = 0

    def add_mentions(self, chunk_id: str, entities: Iterable[str]) -> None:
        changed = False
        for entity in entities:
            key = canonicalize_entity(entity)
            if key and chunk_id not in self._mentions[key]:
                self._mentions[key].add(chunk_id)
                changed = True
        if changed:
            self.revision += 1

    def add_relation(
        self, subject: str, predicate: str, object: str, chunk_id: str
    ) -> None:
        source = canonicalize_entity(subject)
        target = canonicalize_entity(object)
        relation_type = " ".join(predicate.casefold().split())
        if not source or not target or not relation_type or not chunk_id:
            raise ValueError("relation requires entities, predicate, and source chunk")
        edge = EvidenceRelation(source, relation_type, target, chunk_id)
        self._relations.append(edge)
        self._adjacency[source].append(edge)
        self._adjacency[target].append(edge)
        self.add_mentions(chunk_id, (source, target))
        self.revision += 1

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove stale mentions and edges when their source chunk is replaced."""
        changed = False
        for entity in list(self._mentions):
            if chunk_id in self._mentions[entity]:
                self._mentions[entity].remove(chunk_id)
                changed = True
            if not self._mentions[entity]:
                del self._mentions[entity]
        kept = [edge for edge in self._relations if edge.chunk_id != chunk_id]
        if len(kept) != len(self._relations):
            changed = True
            self._relations = kept
            self._adjacency = defaultdict(list)
            for edge in kept:
                self._adjacency[edge.subject].append(edge)
                self._adjacency[edge.object].append(edge)
        if changed:
            self.revision += 1

    def expand(
        self,
        entities: Iterable[str],
        chunks: Mapping[str, ChunkT],
        principal: PrincipalT,
        authorize: Callable[[PrincipalT, ChunkT], bool],
        *,
        max_hops: int = 2,
        max_entities: int = 40,
        max_chunks: int = 20,
    ) -> list[GraphHit]:
        """Return only authorized evidence reached within a bounded traversal.

        Unauthorized edges are removed *before* traversal. This prevents their
        entity names from acting as bridges to other chunks.
        """
        if max_hops not in (1, 2):
            raise ValueError("max_hops must be 1 or 2")
        if max_entities < 1 or max_chunks < 1:
            raise ValueError("graph budgets must be positive")

        allowed: dict[str, ChunkT] = {
            chunk_id: chunk
            for chunk_id, chunk in chunks.items()
            if authorize(principal, chunk)
        }
        # Preserve the linker's relevance order instead of sorting names.
        starts = list(dict.fromkeys(
            key for entity in entities if (key := canonicalize_entity(entity))
        ))
        queue = deque((key, 0, ()) for key in starts[:max_entities])
        seen_entities = set(starts[:max_entities])
        hits: dict[str, GraphHit] = {}

        while queue and len(hits) < max_chunks:
            entity, depth, trail = queue.popleft()
            # Direct query mentions are seeds. Following a relation requires
            # that relation's own evidence; a shared name alone is not a hop.
            if depth == 0:
                for chunk_id in sorted(self._mentions.get(entity, ())):
                    if chunk_id in allowed and chunk_id not in hits:
                        hits[chunk_id] = GraphHit(allowed[chunk_id], depth, trail)
                        if len(hits) >= max_chunks:
                            break
            if depth >= max_hops or len(hits) >= max_chunks:
                continue
            for edge in self._adjacency.get(entity, ()):
                if edge.chunk_id not in allowed:
                    continue
                other = edge.object if edge.subject == entity else edge.subject
                if edge.chunk_id not in hits:
                    hits[edge.chunk_id] = GraphHit(allowed[edge.chunk_id], depth + 1, trail + (edge,))
                if other not in seen_entities and len(seen_entities) < max_entities:
                    seen_entities.add(other)
                    queue.append((other, depth + 1, trail + (edge,)))
                if len(hits) >= max_chunks:
                    break
        return list(hits.values())


def combine_retrieval(
    query: str,
    principal: PrincipalT,
    vector_seeds: Sequence[ChunkT],
    graph: EvidenceGraph,
    chunks: Mapping[str, ChunkT],
    query_entities: Iterable[str],
    authorize: Callable[[PrincipalT, ChunkT], bool],
    *,
    budget: int = 8,
    max_hops: int = 2,
    max_graph_chunks: int | None = None,
    rerank: Callable[[str, Sequence[ChunkT]], Sequence[ChunkT]] | None = None,
) -> list[ChunkT]:
    """Merge vector seeds and graph evidence within a fixed context budget."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if max_graph_chunks is not None and max_graph_chunks < 0:
        raise ValueError("max_graph_chunks must be non-negative")
    candidates: dict[str, ChunkT] = {}
    for chunk in vector_seeds:
        if authorize(principal, chunk):
            candidates.setdefault(chunk.chunk_id, chunk)
    graph_limit = max_graph_chunks if max_graph_chunks is not None else max(budget * 4, budget)
    hits = graph.expand(
        query_entities, chunks, principal, authorize,
        max_hops=max_hops, max_chunks=graph_limit,
    ) if graph_limit else []
    for hit in hits:
        if hit.chunk.chunk_id not in candidates:
            candidates[hit.chunk.chunk_id] = hit.chunk
    ordered = list(candidates.values())
    if rerank is not None:
        proposed = rerank(query, ordered)
        allowed_ids = set(candidates)
        seen: set[str] = set()
        ordered = []
        for chunk in proposed:
            if chunk.chunk_id in allowed_ids and chunk.chunk_id not in seen:
                ordered.append(candidates[chunk.chunk_id])
                seen.add(chunk.chunk_id)
    return ordered[:budget]
