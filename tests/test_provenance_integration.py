"""Integration tests for provenance end-to-end: pipeline → Concept.sources → renderer → JSON."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.distillation.renderer import render_concept
from mindforge.distillation.source_ref import SourceRef
from mindforge.pipeline import MindForgePipeline


def _write_transcript(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_renderer_emits_structured_sources_in_frontmatter():
    c = Concept(
        name="KV Cache", definition="d", explanation="e",
        sources=[SourceRef(
            transcript_path="t.md", transcript_hash="h",
            turn_indices=[4, 7], extracted_at="2025-03-14T11:22:00Z",
        )],
    )
    md = render_concept(c)
    assert 'transcript: "t.md"' in md
    assert "turns: [4, 7]" in md
    assert 'extracted_at: "2025-03-14T11:22:00Z"' in md


def test_renderer_falls_back_to_source_files_when_no_sources():
    c = Concept(
        name="X", definition="d", explanation="e",
        source_files=["legacy.md"],
    )
    md = render_concept(c)
    assert 'sources:\n  - "legacy.md"' in md


def test_pipeline_populates_concept_sources_and_writes_provenance(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    _write_transcript(
        transcripts / "t1.md",
        "Assistant: KV Cache is a mechanism that stores the Key and Value matrices from "
        "the attention computation of previously processed tokens, avoiding redundant "
        "recomputation during autoregressive generation. It trades memory for speed.\n",
    )

    out = tmp_path / "out"
    cfg = MindForgeConfig(transcripts_dir=transcripts, output_dir=out)
    pipeline = MindForgePipeline(cfg)
    pipeline.run()

    manifest = out / "concepts.json"
    assert manifest.exists(), "pipeline must write concepts.json"
    store = ConceptStore.load(manifest)
    concepts = store.all()
    assert concepts, "expected at least one concept"

    # At least one concept must cite its source turn.
    with_sources = [c for c in concepts if c.sources]
    assert with_sources, "no concepts received SourceRef citations"
    ref = with_sources[0].sources[0]
    assert ref.transcript_path.endswith("t1.md")
    assert ref.turn_indices, "turn indices should be non-empty"

    # Provenance JSON must be written.
    prov_files = list((out / "provenance").glob("*.json"))
    assert prov_files, "expected per-concept provenance JSON"
    payload = json.loads(prov_files[0].read_text())
    assert "slug" in payload and "sources" in payload
