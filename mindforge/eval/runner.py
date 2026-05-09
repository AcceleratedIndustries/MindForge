"""Runner: ingest fixtures via pipeline, compute scores, render report."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.eval.corpus import load_corpus
from mindforge.eval.retrieval_tuner import sweep_weights
from mindforge.eval.scorer import score_concepts, score_relationships
from mindforge.graph.builder import KnowledgeGraph
from mindforge.pipeline import MindForgePipeline


def run_eval(fixtures_dir: Path, mode: str = "mock", **llm_kwargs: Any) -> dict[str, Any]:
    """Run the pipeline on a fixture directory and score against ground truth.

    ``mode`` is "mock" (default; deterministic smoke test) or "llm" (real
    LLM, used as the quality gate). For LLM mode, pass ``llm_provider``,
    ``llm_model``, ``llm_base_url``, ``llm_api_key`` via kwargs.
    """
    fixtures = load_corpus(fixtures_dir)
    if not fixtures:
        return {"corpus_size": 0, "fixtures": []}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        cfg_kwargs: dict[str, Any] = {
            "transcripts_dir": fixtures_dir,
            "output_dir": out,
            "llm_provider": "mock" if mode == "mock" else "ollama",
        }
        for k, v in llm_kwargs.items():
            if k.startswith("llm_"):
                cfg_kwargs[k] = v
        cfg = MindForgeConfig(**cfg_kwargs)
        cfg.ensure_dirs()
        MindForgePipeline(cfg).run()

        store = ConceptStore.load(out / "concepts.json")
        actual_concepts = [c.to_dict() for c in store.all()]
        actual_rels: list[dict[str, Any]] = []
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


def run_tune_retrieval(
    output_dir: Path,
    k: int = 5,
    step: float = 0.1,
    baseline: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> dict[str, Any]:
    """Sweep hybrid retrieval weights against a pre-ingested KB.

    Loads the KB at ``output_dir`` (must contain concepts.json + graph),
    synthesizes tag-overlap judgments, sweeps the (k, s, g) grid, and
    returns a report dict suitable for both rendering and JSON archival.
    """
    from mindforge.query.engine import RetrievalWeights

    manifest = output_dir / "concepts.json"
    graph_path = output_dir / "graph" / "knowledge_graph.json"
    if not manifest.exists() or not graph_path.exists():
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "tune-retrieval",
            "error": f"No ingested KB at {output_dir} (need concepts.json + graph)",
            "candidates": [],
        }

    store = ConceptStore.load(manifest)
    graph = KnowledgeGraph.load(graph_path)
    candidates = sweep_weights(store, graph, k=k, step=step)

    baseline_weights = RetrievalWeights(
        keyword=baseline[0], semantic=baseline[1], graph=baseline[2]
    )
    baseline_score = next(
        (
            s
            for w, s in candidates
            if abs(w.keyword - baseline_weights.keyword) < 1e-6
            and abs(w.semantic - baseline_weights.semantic) < 1e-6
            and abs(w.graph - baseline_weights.graph) < 1e-6
        ),
        None,
    )
    winner_weights, winner_score = candidates[0]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "tune-retrieval",
        "k": k,
        "step": step,
        "candidates": [
            {
                "keyword": w.keyword,
                "semantic": w.semantic,
                "graph": w.graph,
                "recall_at_k": s,
            }
            for w, s in candidates
        ],
        "winner": {
            "keyword": winner_weights.keyword,
            "semantic": winner_weights.semantic,
            "graph": winner_weights.graph,
            "recall_at_k": winner_score,
        },
        "baseline": {
            "keyword": baseline_weights.keyword,
            "semantic": baseline_weights.semantic,
            "graph": baseline_weights.graph,
            "recall_at_k": baseline_score,
        },
    }


def render_tune_markdown(report: dict[str, Any]) -> str:
    if "error" in report:
        return f"Tune-retrieval error: {report['error']}\n"
    lines = [
        "Hybrid retrieval weight sweep",
        "=" * 40,
        f"  Timestamp:  {report['timestamp']}",
        f"  k:          {report['k']}",
        f"  step:       {report['step']}",
        "",
        f"  Top 10 by recall@{report['k']}:",
    ]
    for c in report["candidates"][:10]:
        lines.append(
            f"    k={c['keyword']:.1f} s={c['semantic']:.1f} g={c['graph']:.1f}"
            f"  recall@{report['k']}={c['recall_at_k']:.3f}"
        )
    lines.append("")
    w = report["winner"]
    b = report["baseline"]
    lines.append(
        f"  Winner:   k={w['keyword']:.1f} s={w['semantic']:.1f} g={w['graph']:.1f}"
        f"  recall@{report['k']}={w['recall_at_k']:.3f}"
    )
    if b["recall_at_k"] is not None:
        delta = w["recall_at_k"] - b["recall_at_k"]
        lines.append(
            f"  Baseline: k={b['keyword']:.1f} s={b['semantic']:.1f} g={b['graph']:.1f}"
            f"  recall@{report['k']}={b['recall_at_k']:.3f}  (winner-baseline = {delta:+.3f})"
        )
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
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
