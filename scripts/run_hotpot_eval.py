"""Evaluate vector retrieval and graph expansion on a frozen HotpotQA sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rag_graph.access import ACLStore
from rag_graph.baseline import (
    ExtractiveDemoGenerator,
    HashingEmbedder,
    LiteLLMGenerator,
    MemoryIndex,
    SentenceTransformerEmbedder,
    ingest_document,
)
from rag_graph.domain import ACL
from rag_graph.evaluation import EvalRow, evaluate_secure, load_questions, summarize, write_csv
from rag_graph.graph import EvidenceGraph
from rag_graph.graph_extraction import add_cooccurrence_evidence, question_linker
from rag_graph.service import GraphRAGService


def experiment_fingerprint(paths: list[Path]) -> str:
    """Bind checkpoints to the exact code and input bytes used by a run."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("data/hotpot_sample"))
    parser.add_argument("--embedder", choices=["hash", "sentence-transformer"], default="hash")
    parser.add_argument("--model", default="extractive-demo")
    parser.add_argument("--graph", choices=["none", "cooccurrence"], default="none")
    parser.add_argument("--reranker", choices=["none", "lexical"], default="none")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test only; omit for the frozen full sample")
    parser.add_argument("--output", type=Path, default=Path("results/hotpot_vector_baseline.csv"))
    parser.add_argument("--price-file", type=Path, default=Path("config/model_prices_2026-09-27.json"))
    args = parser.parse_args()

    embedder = HashingEmbedder() if args.embedder == "hash" else SentenceTransformerEmbedder()
    generator = ExtractiveDemoGenerator() if args.model == "extractive-demo" else LiteLLMGenerator(args.model)
    index, acls = MemoryIndex(), ACLStore()
    graph = EvidenceGraph() if args.graph == "cooccurrence" else None
    known_titles: list[str] = []
    acl = ACL("benchmark", frozenset({"reader"}), frozenset())
    documents = args.sample_dir / "documents.jsonl"
    for line in documents.read_text().splitlines():
        item = json.loads(line)
        known_titles.append(item["title"])
        chunks = ingest_document(item["document_id"], f'title: {item["title"]}\n\n{item["text"]}', acl, embedder)
        index.upsert(chunks)
        acls.update(chunks[0].document_id, acl)
        if graph is not None:
            for chunk in chunks:
                add_cooccurrence_evidence(graph, chunk, item["title"])

    def lexical_rerank(question, chunks):
        terms = set(question.casefold().split())
        return sorted(chunks, key=lambda chunk: len(terms & set(chunk.text.casefold().split())), reverse=True)

    service = GraphRAGService(
        index=index, acl_store=acls, embedder=embedder,
        generator=generator,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        graph=graph,
        question_entities=(lambda question: question_linker(question, known_titles)) if graph is not None else None,
        reranker=lexical_rerank if args.reranker == "lexical" else None,
    )
    price_data = json.loads(args.price_file.read_text())
    model_prices = {
        model: (price["input_per_million"], price["output_per_million"])
        for model, price in price_data["models"].items()
    }
    questions = load_questions(args.sample_dir / "questions.jsonl")
    if args.limit is not None:
        questions = questions[: args.limit]
    code_root = Path(__file__).resolve().parents[1]
    fingerprint = experiment_fingerprint([
        *sorted((code_root / "src/rag_graph").glob("*.py")),
        Path(__file__),
        args.sample_dir / "documents.jsonl",
        args.sample_dir / "questions.jsonl",
        args.sample_dir / "manifest.json",
        args.price_file,
    ])
    config = {"embedder": args.embedder, "model": args.model, "graph": args.graph, "reranker": args.reranker, "top_k": args.top_k, "candidate_pool": args.candidate_pool, "limit": args.limit, "price_date": price_data["price_date"], "content_sha256": fingerprint}
    manifest = json.loads((args.sample_dir / "manifest.json").read_text())
    checkpoint_path = args.output.with_suffix(".checkpoint.json")
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint["configuration"] != config or checkpoint["question_ids_sha256"] != manifest["question_ids_sha256"]:
            raise ValueError("checkpoint configuration differs from this evaluation")
        rows = [EvalRow(**row) for row in checkpoint["rows"]]
    else:
        rows = []
    completed = {row.question_id for row in rows}
    for item in questions:
        if item.question_id in completed:
            continue
        row = evaluate_secure(service, [item], model_prices=model_prices if args.model in model_prices else None)[0]
        rows.append(row)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "configuration": config,
            "question_ids_sha256": manifest["question_ids_sha256"],
            "rows": [asdict(value) for value in rows],
        }))
        temporary.replace(checkpoint_path)
        print(f"Completed {len(rows)}/{len(questions)}: {item.question_id}", flush=True)
    write_csv(rows, args.output)
    result = {
        "configuration": config,
        "manifest": manifest,
        "overall": summarize(rows),
        "bridge": summarize([row for row in rows if row.split == "bridge"]) if any(row.split == "bridge" for row in rows) else None,
        "comparison": summarize([row for row in rows if row.split == "comparison"]) if any(row.split == "comparison" for row in rows) else None,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Question-level results: {args.output}")


if __name__ == "__main__":
    main()
