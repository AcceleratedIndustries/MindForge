"""Pipeline orchestrator: ties all MindForge components together.

This is the main entry point for running the full ingestion-to-query pipeline.
Designed for both batch and incremental processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.distillation.deduplicator import deduplicate_concepts
from mindforge.distillation.renderer import write_all_concepts
from mindforge.embeddings.index import EmbeddingIndex
from mindforge.graph.builder import KnowledgeGraph
from mindforge.ingestion.chunker import Chunk, chunk_turns
from mindforge.ingestion.extractor import RawConcept, extract_concepts
from mindforge.ingestion.parser import parse_all_transcripts
from mindforge.linking.linker import detect_links
from mindforge.llm.client import LLMClient, LLMConfig
from mindforge.llm.distiller import distill_all_smart
from mindforge.llm.extractor import extract_concepts_llm
from mindforge.query.engine import QueryEngine


def _write_all_provenance(concepts: list[Concept], provenance_dir: Path) -> int:
    """Write one provenance JSON per concept that has sources. Returns file count."""
    count = 0
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        if not concept.sources:
            continue
        path = provenance_dir / f"{concept.slug}.json"
        path.write_text(
            json.dumps(
                {
                    "slug": concept.slug,
                    "name": concept.name,
                    "sources": [s.to_dict() for s in concept.sources],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        count += 1
    return count


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    concepts_extracted: int
    concepts_after_dedup: int
    concept_files_written: int
    edges_in_graph: int
    embeddings_built: bool
    extraction_method: str = "heuristic"

    def summary(self) -> str:
        lines = [
            "MindForge Pipeline Complete",
            "=" * 40,
            f"  Extraction method:       {self.extraction_method}",
            f"  Concepts extracted:      {self.concepts_extracted}",
            f"  After deduplication:     {self.concepts_after_dedup}",
            f"  Markdown files written:  {self.concept_files_written}",
            f"  Graph edges:             {self.edges_in_graph}",
            f"  Embeddings built:        "
            f"{'yes' if self.embeddings_built else 'no (optional deps not installed)'}",
        ]
        return "\n".join(lines)


class MindForgePipeline:
    """Orchestrates the full MindForge knowledge extraction pipeline."""

    def __init__(self, config: MindForgeConfig | None = None) -> None:
        self.config = config or MindForgeConfig()
        self.store = ConceptStore()
        self.graph: KnowledgeGraph | None = None
        self.embedding_index: EmbeddingIndex | None = None
        self.query_engine: QueryEngine | None = None

    def run(self) -> PipelineResult:
        """Execute the full pipeline: ingest → extract → distill → link → graph → index."""
        self.config.ensure_dirs()

        # === Stage 1: Ingestion ===
        print("[1/6] Parsing transcripts...")
        transcripts = parse_all_transcripts(self.config.transcripts_dir)
        if not transcripts:
            print(f"  No transcripts found in {self.config.transcripts_dir}")
            return PipelineResult(0, 0, 0, 0, False)

        total_turns = sum(len(t.turns) for t in transcripts)
        print(f"  Found {len(transcripts)} file(s), {total_turns} turns")

        # === Stage 2: Chunking & Extraction ===
        print("[2/6] Chunking and extracting concepts...")
        all_chunks = []
        for transcript in transcripts:
            # Focus on assistant turns (where knowledge lives)
            chunks = chunk_turns(transcript.assistant_turns)
            all_chunks.extend(chunks)
        print(f"  Generated {len(all_chunks)} semantic chunks")

        extraction_method = "heuristic"
        raw_concepts: list[RawConcept] = []

        if self.config.use_llm:
            raw_concepts, extraction_method = self._extract_with_llm(all_chunks)
        else:
            raw_concepts = extract_concepts(all_chunks)

        print(f"  Extracted {len(raw_concepts)} candidate concepts")

        # === Stage 3: Deduplication ===
        print("[3/6] Deduplicating concepts...")
        deduped = deduplicate_concepts(
            raw_concepts,
            similarity_threshold=self.config.similarity_threshold,
        )
        print(f"  {len(raw_concepts)} → {len(deduped)} after deduplication")

        # === Stage 4: Distillation ===
        print("[4/6] Distilling concepts...")
        # Build a chunk map so the distiller can attach SourceRef citations.
        chunk_map = {c.id: c for c in all_chunks}
        # Use smart distiller that handles both LLM and heuristic concepts
        concepts = distill_all_smart(deduped, chunk_map)

        # Add to store (handles merging with existing concepts)
        for concept in concepts:
            self.store.add(concept)

        # Stamp last_reinforced_at for hygiene bookkeeping.
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        for concept in self.store.all():
            concept.last_reinforced_at = now_iso

        # Save manifest
        manifest_path = self.config.output_dir / "concepts.json"
        self.store.save(manifest_path)

        # Write per-concept provenance JSON files.
        _write_all_provenance(self.store.all(), self.config.provenance_dir)

        # === Stage 5: Linking ===
        print("[5/6] Detecting links and relationships...")
        detect_links(self.store, self.config.link_confidence_threshold)

        # Write Markdown files
        written = write_all_concepts(self.store.all(), self.config.concepts_dir)
        print(f"  Wrote {len(written)} concept files to {self.config.concepts_dir}")

        # Build graph
        self.graph = KnowledgeGraph.from_store(self.store)
        graph_path = self.config.graph_dir / "knowledge_graph.json"
        self.graph.save(graph_path)
        stats = self.graph.stats()
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

        # === Stage 6: Embeddings (optional) ===
        print("[6/6] Building embeddings index...")
        embeddings_built = False
        if self.config.use_embeddings:
            self.embedding_index = EmbeddingIndex(self.config.embedding_model)
            if self.embedding_index.available:
                self.embedding_index.build(self.store.all())
                self.embedding_index.save(self.config.embeddings_dir)
                embeddings_built = True
                print("  Embeddings index built and saved")
            else:
                print("  Skipped: install mindforge[embeddings] for semantic search")
        else:
            print("  Skipped: pass --embeddings to enable")

        # Initialize query engine
        self.query_engine = QueryEngine(
            self.store,
            self.graph,
            self.embedding_index,
        )

        # Save updated manifest (now with links)
        self.store.save(manifest_path)

        return PipelineResult(
            concepts_extracted=len(raw_concepts),
            concepts_after_dedup=len(deduped),
            concept_files_written=len(written),
            edges_in_graph=stats["edges"],
            embeddings_built=embeddings_built,
            extraction_method=extraction_method,
        )

    def query(self, question: str, top_k: int = 5) -> str:
        """Query the knowledge base. Pipeline must have been run first."""
        if self.query_engine is None:
            # Try to load from disk
            self._load_state()
        if self.query_engine is None:
            return "Error: No knowledge base found. Run the pipeline first."

        results = self.query_engine.search(question, top_k=top_k)
        return self.query_engine.format_results(results)

    def _extract_with_llm(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[RawConcept], str]:
        """Attempt LLM extraction, falling back to heuristic if unavailable.

        Returns (concepts, extraction_method_name).
        """
        llm_config = LLMConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )
        client = LLMClient(llm_config)

        if not client.available:
            print(f"  LLM server not reachable ({llm_config.base_url})")
            print("  Falling back to heuristic extraction")
            return extract_concepts(chunks), "heuristic (LLM unavailable)"

        print(f"  Using LLM: {llm_config.provider}/{llm_config.model}")
        llm_concepts, stats = extract_concepts_llm(chunks, client)

        if stats.parse_failures > 0:
            print(f"  Warning: {stats.parse_failures} LLM parse failure(s)")

        # Also run heuristic extraction and merge (LLM may miss things
        # that pattern matching catches, and vice versa)
        heuristic_concepts = extract_concepts(chunks)
        print(
            f"  LLM extracted {len(llm_concepts)} concepts, "
            f"heuristic found {len(heuristic_concepts)}"
        )

        # Merge: LLM concepts take priority, then add unique heuristic ones
        seen_names = {c.name.lower() for c in llm_concepts}
        merged = list(llm_concepts)
        for hc in heuristic_concepts:
            if hc.name.lower() not in seen_names:
                seen_names.add(hc.name.lower())
                merged.append(hc)

        method = f"llm ({llm_config.provider}/{llm_config.model}) + heuristic"
        return merged, method

    def _load_state(self) -> None:
        """Load previously built state from disk."""
        manifest = self.config.output_dir / "concepts.json"
        graph_file = self.config.graph_dir / "knowledge_graph.json"

        if manifest.exists():
            self.store = ConceptStore.load(manifest)

        graph = None
        if graph_file.exists():
            graph = KnowledgeGraph.load(graph_file)
            self.graph = graph

        index = None
        if self.config.use_embeddings:
            index = EmbeddingIndex.load(self.config.embeddings_dir)
            if index.available:
                self.embedding_index = index

        self.query_engine = QueryEngine(self.store, graph, index)
