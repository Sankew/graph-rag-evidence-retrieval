import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_hotpot_sample import build_sample
from rag_graph.evaluation import load_questions


def test_hotpot_sample_is_deterministic_and_keeps_support_labels(tmp_path: Path):
    raw = []
    for kind in ("bridge", "comparison"):
        for n in range(2):
            raw.append({
                "_id": f"{kind}-{n}", "type": kind,
                "question": "Who wrote it?", "answer": "Ada",
                "context": [[f"Title {kind} {n}", ["Ada wrote it."]], [f"Distractor {n}", ["Other text."]]],
                "supporting_facts": [[f"Title {kind} {n}", 0]],
            })
    source = tmp_path / "source.json"
    source.write_text(json.dumps(raw))
    first = build_sample(source, tmp_path / "first", count=4)
    second = build_sample(source, tmp_path / "second", count=4)
    assert first["question_ids_sha256"] == second["question_ids_sha256"]
    assert (tmp_path / "first/questions.jsonl").read_bytes() == (tmp_path / "second/questions.jsonl").read_bytes()
    documents = [json.loads(line) for line in (tmp_path / "first/documents.jsonl").read_text().splitlines()]
    assert all(set(document) == {"document_id", "title", "text"} for document in documents)
    questions = load_questions(tmp_path / "first/questions.jsonl")
    assert len(questions) == 4
    assert all(q.supporting_document_ids[0].startswith("benchmark:") for q in questions)
