"""Runner: ingest fixtures via pipeline, compute scores, render report."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.eval.corpus import load_corpus
from mindforge.eval.scorer import score_concepts, score_relationships
from mindforge.pipeline import MindForgePipeline


def run_eval(fixtures_dir: Path, mode: str = "heuristic", **llm_kwargs) -> dict:
    """Run the pipeline on a fixture directory and score against ground truth.

    ``mode`` is "heuristic" (default) or "llm". For LLM mode, pass
    ``llm_provider``, ``llm_model``, ``llm_base_url``, ``llm_api_key`` via kwargs.
    """
    fixtures = load_corpus(fixtures_dir)
    if not fixtures:
        return {"corpus_size": 0, "fixtures": []}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        cfg_kwargs = {
            "transcripts_dir": fixtures_dir,
            "output_dir": out,
            "use_llm": mode == "llm",
        }
        for k, v in llm_kwargs.items():
            if k.startswith("llm_"):
                cfg_kwargs[k] = v
        cfg = MindForgeConfig(**cfg_kwargs)
        cfg.ensure_dirs()
        MindForgePipeline(cfg).run()

        store = ConceptStore.load(out / "concepts.json")
        actual_concepts = [c.to_dict() for c in store.all()]
        actual_rels: list[dict] = []
        for c in store.all():
            for r in c.relationships:
                actual_rels.append(r.to_dict())

    expected_concepts = [e for f in fixtures for e in f.expected_concepts]
    expected_rels = [r for f in fixtures for r in f.expected_relationships]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "corpus_size": len(fixtures),
        "concepts": score_concepts(expected_concepts, actual_concepts),
        "relationships": score_relationships(expected_rels, actual_rels),
    }


def render_markdown(report: dict) -> str:
    if report.get("corpus_size", 0) == 0:
        return "MindForge Evaluation Report\n\n(no fixtures)\n"
    c = report["concepts"]
    r = report["relationships"]
    lines = [
        "MindForge Evaluation Report",
        "=" * 40,
        f"  Mode:       {report['mode']}",
        f"  Timestamp:  {report['timestamp']}",
        f"  Corpus:     {report['corpus_size']} fixtures",
        "",
        "  Concepts",
        f"    Expected:          {c['expected']}",
        f"    Extracted:         {c['extracted']}",
        f"    Matched:           {c['matched']}",
        f"    Recall:            {c['recall']}",
        f"    Precision:         {c['precision']}",
        f"    Phrase grounding:  {c['phrase_grounding']}",
        "",
        "  Relationships",
        f"    Expected:          {r['expected']}",
        f"    Found:             {r['found']}",
        f"    Matched:           {r['matched']}",
        f"    Recall:            {r['recall']}",
        f"    Type accuracy:     {r['type_accuracy']}",
    ]
    return "\n".join(lines) + "\n"
