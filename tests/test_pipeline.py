"""Tests for the full MindForge pipeline."""

import json
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


class TestPipeline:
    def _create_transcript(self, tmp_path: Path) -> Path:
        """Create a minimal test transcript."""
        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()
        transcript = transcripts_dir / "test.md"
        transcript.write_text(
            "Human: What is a vector database?\n\n"
            "Assistant: ## Vector Database\n\n"
            "Vector Database is a specialized database system optimized for storing "
            "and querying high-dimensional vector data. It enables efficient similarity "
            "search using algorithms like HNSW.\n\n"
            "## Similarity Search\n\n"
            "Similarity Search is a technique that finds items closest to a query point "
            "in vector space. It relies on distance metrics like cosine similarity or "
            "Euclidean distance.\n\n"
            "Vector Database enables fast Similarity Search at scale.\n"
        )
        return transcripts_dir

    def test_full_pipeline(self, tmp_path):
        transcripts_dir = self._create_transcript(tmp_path)
        output_dir = tmp_path / "output"

        config = MindForgeConfig(
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            llm_provider="mock",
        )
        pipeline = MindForgePipeline(config)
        result = pipeline.run()

        assert result.concepts_extracted > 0
        assert result.concept_files_written > 0

        # Check that concept files were created
        concept_files = list(config.concepts_dir.glob("*.md"))
        assert len(concept_files) > 0

        # Check that graph was created
        graph_file = config.graph_dir / "knowledge_graph.json"
        assert graph_file.exists()

        # Check manifest
        manifest = output_dir / "concepts.json"
        assert manifest.exists()

    def test_pipeline_with_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        config = MindForgeConfig(
            transcripts_dir=empty_dir,
            output_dir=output_dir,
        )
        pipeline = MindForgePipeline(config)
        result = pipeline.run()

        assert result.concepts_extracted == 0

    def test_query_after_pipeline(self, tmp_path):
        transcripts_dir = self._create_transcript(tmp_path)
        output_dir = tmp_path / "output"

        config = MindForgeConfig(
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            llm_provider="mock",
        )
        pipeline = MindForgePipeline(config)
        pipeline.run()

        result = pipeline.query("What is a vector database?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_result_summary(self, tmp_path):
        transcripts_dir = self._create_transcript(tmp_path)
        output_dir = tmp_path / "output"

        config = MindForgeConfig(
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            llm_provider="mock",
        )
        pipeline = MindForgePipeline(config)
        result = pipeline.run()

        summary = result.summary()
        assert "MindForge Pipeline Complete" in summary
        assert "Concepts extracted" in summary


class TestProvenanceAccuracy:
    """End-to-end test that per-concept provenance narrows turn_indices to
    supporting chunks, not the full batch."""

    def test_turn_indices_narrowed_to_supporting_turns(self, tmp_path: Path) -> None:
        # Build a transcript where 'KV Cache' is mentioned in exactly one
        # assistant turn, with many other turns of unrelated content. After
        # ingest, the concept's SourceRef should list only the supporting
        # turn(s), not every turn the LLM saw in the same batch.

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        # Mock LLM produces deterministic title-case extraction (cap of 3
        # multi-word title-case phrases per batch). 'KV Cache' is the only
        # multi-word title-case phrase in the concatenated batch, so it gets
        # extracted. Only the first assistant turn (index 1) mentions 'KV
        # Cache'; the other three mention 'cache' (lowercase) so the
        # token-bounded grounding match for 'KV Cache' rejects them.
        # Attribution should narrow to turn 1 only, not the full batch's
        # [1, 3, 5, 7].
        transcript = transcripts / "session.md"
        transcript.write_text(
            "Human: hi\n\n"
            "Assistant: We use KV Cache for inference speedup.\n\n"
            "Human: ok\n\n"
            "Assistant: then we discussed cache hits in the buffer.\n\n"
            "Human: continue\n\n"
            "Assistant: eventually, cache eviction strategies came up.\n\n"
            "Human: cool\n\n"
            "Assistant: wrapping up with cache invalidation concerns.\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
        )
        pipe = MindForgePipeline(cfg)
        result = pipe.run()
        assert result.concept_files_written >= 1

        # Load concepts.json and find the KV Cache concept.
        store = json.loads((out / "concepts.json").read_text(encoding="utf-8"))
        kv = store.get("kv-cache")
        assert kv is not None, "expected 'kv-cache' concept (mock client always emits title-case)"

        sources = kv.get("sources", [])
        assert len(sources) >= 1
        # Crucial assertion: turn_indices contains the supporting turn(s) only.
        # The transcript has 4 assistant turns (indices depend on the parser
        # numbering; check that turn_indices is NOT the full set of assistant
        # turns).
        for src in sources:
            assert isinstance(src["turn_indices"], list)
            assert len(src["turn_indices"]) >= 1
            # The full batch would have included all 4 assistant turns
            # (indices roughly [1, 3, 5, 7] depending on parser conventions).
            # The narrow result should have FEWER than 4 entries.
            assert len(src["turn_indices"]) < 4, (
                f"turn_indices too broad: {src['turn_indices']} - should "
                "contain only the supporting turn(s), not every batch turn"
            )

    def test_unsupported_chunks_not_listed(self, tmp_path: Path) -> None:
        # Two-transcript fixture: transcript A mentions 'KV Cache', transcript
        # B does not. The KV Cache concept's source_files / sources should
        # contain transcript A only.

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        (transcripts / "a.md").write_text(
            "Human: q\n\nAssistant: We use KV Cache for inference.\n",
            encoding="utf-8",
        )
        (transcripts / "b.md").write_text(
            "Human: q\n\nAssistant: Today's lunch was pasta.\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
        )
        pipe = MindForgePipeline(cfg)
        pipe.run()

        store = json.loads((out / "concepts.json").read_text(encoding="utf-8"))
        kv = store.get("kv-cache")
        assert kv is not None
        # The concept's sources should reference only a.md (the supporting
        # transcript), not b.md.
        source_paths = {src["transcript_path"] for src in kv["sources"]}
        assert any(str(transcripts / "a.md") in p for p in source_paths)
        assert not any(str(transcripts / "b.md") in p for p in source_paths)
