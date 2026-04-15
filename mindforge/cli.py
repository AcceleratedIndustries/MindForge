"""Command-line interface for MindForge.

Usage:
    mindforge ingest [--input DIR] [--output DIR] [--embeddings]
    mindforge query "your question here"
    mindforge stats
    mindforge --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindforge",
        description="MindForge: Transform AI conversations into structured knowledge.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest transcripts and build the knowledge base",
    )
    ingest.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("examples/transcripts"),
        help="Directory containing transcript files (default: examples/transcripts)",
    )
    ingest.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    ingest.add_argument(
        "--embeddings",
        action="store_true",
        help="Build embeddings index for semantic search (requires optional deps)",
    )
    ingest.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for deduplication (default: 0.75)",
    )

    # --- query ---
    query = subparsers.add_parser(
        "query",
        help="Query the knowledge base",
    )
    query.add_argument(
        "question",
        help="Natural language question to search for",
    )
    query.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    query.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory to load knowledge base from",
    )
    query.add_argument(
        "--embeddings",
        action="store_true",
        help="Use embeddings for semantic search",
    )

    # --- stats ---
    stats = subparsers.add_parser(
        "stats",
        help="Show knowledge base statistics",
    )
    stats.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline."""
    config = MindForgeConfig(
        transcripts_dir=args.input,
        output_dir=args.output,
        use_embeddings=args.embeddings,
        similarity_threshold=args.similarity_threshold,
    )

    print(f"MindForge v0.1.0")
    print(f"Input:  {config.transcripts_dir.resolve()}")
    print(f"Output: {config.output_dir.resolve()}")
    print()

    pipeline = MindForgePipeline(config)
    result = pipeline.run()

    print()
    print(result.summary())
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the knowledge base."""
    config = MindForgeConfig(
        output_dir=args.output,
        use_embeddings=args.embeddings,
    )

    pipeline = MindForgePipeline(config)
    output = pipeline.query(args.question, top_k=args.top_k)
    print(output)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show knowledge base statistics."""
    from mindforge.distillation.concept import ConceptStore
    from mindforge.graph.builder import KnowledgeGraph

    config = MindForgeConfig(output_dir=args.output)

    manifest = config.output_dir / "concepts.json"
    if not manifest.exists():
        print("No knowledge base found. Run 'mindforge ingest' first.")
        return 1

    store = ConceptStore.load(manifest)
    concepts = store.all()

    print(f"MindForge Knowledge Base Statistics")
    print(f"{'=' * 40}")
    print(f"  Total concepts:    {len(concepts)}")

    if concepts:
        avg_confidence = sum(c.confidence for c in concepts) / len(concepts)
        total_insights = sum(len(c.insights) for c in concepts)
        total_links = sum(len(c.links) for c in concepts)
        print(f"  Avg confidence:    {avg_confidence:.2f}")
        print(f"  Total insights:    {total_insights}")
        print(f"  Total links:       {total_links}")
        print()
        print("  Concepts:")
        for c in sorted(concepts, key=lambda x: x.confidence, reverse=True):
            print(f"    [{c.confidence:.2f}] {c.name}")

    graph_path = config.graph_dir / "knowledge_graph.json"
    if graph_path.exists():
        graph = KnowledgeGraph.load(graph_path)
        stats = graph.stats()
        print()
        print(f"  Graph:")
        print(f"    Nodes:     {stats['nodes']}")
        print(f"    Edges:     {stats['edges']}")
        print(f"    Clusters:  {stats['clusters']}")
        if "density" in stats:
            print(f"    Density:   {stats['density']}")

        top = graph.central_concepts(top_n=5)
        if top:
            print()
            print("  Most Central Concepts:")
            for slug, centrality in top:
                concept = store.get(slug)
                name = concept.name if concept else slug
                print(f"    {name}: {centrality:.3f}")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
