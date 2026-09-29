"""Freeze a balanced HotpotQA distractor sample without changing its labels.

Download the official dev JSON yourself; this script keeps the large licensed
source file outside Git and writes a deterministic, auditable local sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def document_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:36]
    digest = hashlib.sha256(title.encode()).hexdigest()[:10]
    return f"{slug}-{digest}"


def build_sample(input_path: Path, output_dir: Path, count: int = 150) -> dict[str, object]:
    if count < 2:
        raise ValueError("count must be at least 2")
    source = input_path.read_bytes()
    if input_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        questions = [_parquet_to_original(row) for row in pq.read_table(input_path).to_pylist()]
        source_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fpr%2F9/distractor/validation-00000-of-00001.parquet"
    else:
        questions = json.loads(source)
        source_url = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    if not isinstance(questions, list):
        raise ValueError("HotpotQA input must be a JSON array")
    by_type: dict[str, list[dict]] = {"bridge": [], "comparison": []}
    for item in questions:
        if item.get("type") in by_type:
            by_type[item["type"]].append(item)
    for group in by_type.values():
        group.sort(key=lambda item: hashlib.sha256(item["_id"].encode()).hexdigest())
    bridge_count = count // 2
    selected = by_type["bridge"][:bridge_count] + by_type["comparison"][:count - bridge_count]
    if len(selected) != count:
        raise ValueError("source lacks enough bridge or comparison examples")
    selected.sort(key=lambda item: item["_id"])

    documents: dict[str, dict[str, object]] = {}
    eval_cases: list[dict[str, object]] = []
    for item in selected:
        titles: dict[str, str] = {}
        for title, sentences in item["context"]:
            doc_id = document_id(title)
            titles[title] = doc_id
            text = " ".join(sentences)
            prior = documents.get(doc_id)
            if prior is not None and prior["text"] != text:
                raise ValueError(f"conflicting text for title {title!r}")
            documents[doc_id] = {"document_id": doc_id, "title": title, "text": text}
        supporting_titles = {title for title, _ in item["supporting_facts"]}
        if not supporting_titles <= titles.keys():
            raise ValueError(f"supporting title absent from context for {item['_id']}")
        eval_cases.append({
            "question_id": item["_id"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "supporting_document_ids": sorted(f"benchmark:{titles[title]}" for title in supporting_titles),
            "split": item["type"],
            "user_id": "benchmark-reader",
            "tenant_id": "benchmark",
            "roles": ["reader"],
            "clearance": 0,
            "supporting_facts": [
                {"document_id": f"benchmark:{titles[title]}", "sentence_index": index}
                for title, index in item["supporting_facts"]
            ],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "documents.jsonl").open("w") as out:
        for item in sorted(documents.values(), key=lambda row: row["document_id"]):
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "questions.jsonl").open("w") as out:
        for item in eval_cases:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "source": source_url,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "dataset_license": "CC BY-SA 4.0",
        "selection": "SHA-256 question ID order, half bridge and half comparison",
        "question_count": len(eval_cases),
        "document_count": len(documents),
        "question_ids_sha256": hashlib.sha256("\n".join(row["question_id"] for row in eval_cases).encode()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _parquet_to_original(row: dict[str, object]) -> dict[str, object]:
    context = row["context"]
    support = row["supporting_facts"]
    return {
        "_id": row["id"],
        "type": row["type"],
        "question": row["question"],
        "answer": row["answer"],
        "context": list(zip(context["title"], context["sentences"])),
        "supporting_facts": list(zip(support["title"], support["sent_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="official development JSON or HotpotQA validation Parquet")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(build_sample(args.input, args.output_dir, args.count), indent=2))


if __name__ == "__main__":
    main()
