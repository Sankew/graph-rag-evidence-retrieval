"""Frozen-question evaluation with transparent retrieval and answer metrics."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .baseline import BaselinePipeline
from .domain import Principal


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalized_tokens(prediction) == normalized_tokens(reference))


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    predicted, expected = Counter(normalized_tokens(prediction)), Counter(normalized_tokens(reference))
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((predicted & expected).values())
    if not common:
        return 0.0
    precision = common / sum(predicted.values())
    recall = common / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question: str
    reference_answer: str
    supporting_document_ids: tuple[str, ...]
    split: str = "eval"
    user_id: str = ""
    tenant_id: str = ""
    roles: tuple[str, ...] = ()
    clearance: int = 0


@dataclass(frozen=True)
class EvalRow:
    question_id: str
    split: str
    prediction: str
    reference_answer: str
    exact_match: float
    answer_f1: float
    supporting_document_recall: float | None
    supporting_document_all_hit: float | None
    retrieved_document_ids: str
    cited_document_ids: str
    latency_ms: float
    model_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    citation_precision: float | None = None
    abstention_correct: float | None = None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] * (high - rank) + ordered[high] * (rank - low) if low != high else ordered[low]


def evaluate(
    pipeline: BaselinePipeline,
    questions: Iterable[EvalQuestion],
    *,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
    model_prices: dict[str, tuple[float, float]] | None = None,
) -> list[EvalRow]:
    """Score a baseline pipeline on the same question schema used by the service."""
    rows: list[EvalRow] = []
    for item in questions:
        result = pipeline.answer(item.question)
        rows.append(_score(item, result, input_price_per_million, output_price_per_million, model_prices))
    return rows


def evaluate_secure(
    service: object,
    questions: Iterable[EvalQuestion],
    *,
    model_prices: dict[str, tuple[float, float]] | None = None,
) -> list[EvalRow]:
    """Score the authenticated vector or graph service."""
    rows: list[EvalRow] = []
    for item in questions:
        if not item.user_id or not item.tenant_id:
            raise ValueError(f"question {item.question_id} needs an eval persona")
        principal = Principal(item.user_id, item.tenant_id, frozenset(item.roles), item.clearance)
        result = service.answer(principal, item.question)
        rows.append(_score(item, result, 0.0, 0.0, model_prices))
    return rows


def _score(
    item: EvalQuestion,
    result: object,
    input_price: float,
    output_price: float,
    model_prices: dict[str, tuple[float, float]] | None,
) -> EvalRow:
    answer = result.answer
    retrieved = {hit.chunk.document_id for hit in result.retrieved}
    expected = set(item.supporting_document_ids)
    cited = {citation.document_id for citation in answer.citations}
    if model_prices is not None:
        input_price, output_price = model_prices[answer.model_id]
    input_tokens = answer.input_tokens
    output_tokens = answer.output_tokens
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    abstained = not cited and ("don't know" in answer.text.casefold() or "cannot answer" in answer.text.casefold())
    return EvalRow(
        item.question_id,
        item.split,
        answer.text,
        item.reference_answer,
        exact_match(answer.text, item.reference_answer),
        token_f1(answer.text, item.reference_answer),
        len(expected & retrieved) / len(expected) if expected else None,
        float(expected <= retrieved) if expected else None,
        ",".join(sorted(retrieved)),
        ",".join(sorted(cited)),
        result.latency_ms,
        answer.model_id,
        input_tokens,
        output_tokens,
        cost,
        len(cited & expected) / len(cited) if cited and expected else None,
        float(abstained) if not expected else None,
    )


def summarize(rows: list[EvalRow]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    answerable = [row for row in rows if row.supporting_document_recall is not None]
    unanswerable = [row for row in rows if row.abstention_correct is not None]
    cited = [row for row in rows if row.citation_precision is not None]
    return {
        "questions": len(rows),
        "exact_match": mean(row.exact_match for row in rows),
        "answer_f1": mean(row.answer_f1 for row in rows),
        "supporting_document_recall": mean(row.supporting_document_recall for row in answerable) if answerable else 0.0,
        "supporting_document_all_hit": mean(row.supporting_document_all_hit for row in answerable) if answerable else 0.0,
        "citation_precision": mean(row.citation_precision for row in cited) if cited else 0.0,
        "abstention_accuracy": mean(row.abstention_correct for row in unanswerable) if unanswerable else 0.0,
        "latency_p50_ms": percentile([row.latency_ms for row in rows], 0.5),
        "latency_p95_ms": percentile([row.latency_ms for row in rows], 0.95),
        "estimated_cost_usd": sum(row.estimated_cost_usd for row in rows),
    }


def load_questions(path: str | Path) -> list[EvalQuestion]:
    questions = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            data["supporting_document_ids"] = tuple(data["supporting_document_ids"])
            data.pop("supporting_facts", None)
            if "roles" in data:
                data["roles"] = tuple(data["roles"])
            questions.append(EvalQuestion(**data))
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("question IDs must be unique")
    return questions


def write_csv(rows: list[EvalRow], path: str | Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty evaluation")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
