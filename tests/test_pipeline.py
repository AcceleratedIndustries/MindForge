"""Tests for the full MindForge pipeline."""

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
        )
        pipeline = MindForgePipeline(config)
        result = pipeline.run()

        summary = result.summary()
        assert "MindForge Pipeline Complete" in summary
        assert "Concepts extracted" in summary
