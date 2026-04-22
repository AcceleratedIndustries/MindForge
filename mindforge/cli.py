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
from typing import Any

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import Concept, ConceptStore
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
        "--input",
        "-i",
        type=Path,
        default=Path("examples/transcripts"),
        help="Directory containing transcript files (default: examples/transcripts)",
    )
    ingest.add_argument(
        "--output",
        "-o",
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
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview diff against the existing KB without writing anything",
    )

    # LLM extraction options
    llm_group = ingest.add_argument_group("LLM extraction")
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-assisted concept extraction (requires Ollama or API)",
    )
    llm_group.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    llm_group.add_argument(
        "--llm-model",
        default="llama3.2",
        help="LLM model name (default: llama3.2)",
    )
    llm_group.add_argument(
        "--llm-base-url",
        default="",
        help="LLM API base URL (default: auto-detect from provider)",
    )
    llm_group.add_argument(
        "--llm-api-key",
        default="",
        help="API key for OpenAI-compatible providers",
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
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    query.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory to load knowledge base from",
    )
    query.add_argument(
        "--embeddings",
        action="store_true",
        help="Use embeddings for semantic search",
    )
    query.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Restrict results to concepts carrying this tag",
    )
    query.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Drop results below this confidence floor",
    )
    query.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp — keep only concepts reinforced on/after this date",
    )

    # --- list ---
    lst = subparsers.add_parser(
        "list",
        help="List concepts with optional filters",
    )
    lst.add_argument("--tag", type=str, default=None)
    lst.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp — keep only concepts reinforced on/after this date",
    )
    lst.add_argument("--min-confidence", type=float, default=None)
    lst.add_argument("--limit", type=int, default=None)
    lst.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
    )

    # --- stats ---
    stats = subparsers.add_parser(
        "stats",
        help="Show knowledge base statistics",
    )
    stats.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    # --- diff ---
    diff = subparsers.add_parser(
        "diff",
        help="Show concept changes since the previous ingest",
    )
    diff.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp — compare against the snapshot at/after this time",
    )
    diff.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
    )

    # --- mcp ---
    mcp = subparsers.add_parser(
        "mcp",
        help="Start the MCP (Model Context Protocol) server for AI agent access",
    )
    mcp.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory containing the knowledge base (default: output)",
    )

    # --- review ---
    review = subparsers.add_parser(
        "review",
        help="Walk the hygiene review queue (conflicts, stale, orphans)",
    )
    review.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    # --- eval ---
    ev = subparsers.add_parser(
        "eval",
        help="Run the evaluation harness against a fixture corpus",
    )
    ev.add_argument(
        "--fixtures",
        type=Path,
        default=Path("eval/fixtures"),
        help="Fixture directory containing *.md + *.gt.yaml pairs",
    )
    ev.add_argument(
        "--reports",
        type=Path,
        default=Path("eval/reports"),
        help="Where to write JSON reports (default: eval/reports)",
    )
    ev.add_argument(
        "--mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Extraction mode (default: heuristic)",
    )

    # --- show ---
    show = subparsers.add_parser(
        "show",
        help="Show a single concept by slug",
    )
    show.add_argument("slug", help="Concept slug to display")
    show.add_argument(
        "--sources",
        action="store_true",
        help="Print source citations (transcript paths + turn indices)",
    )
    show.add_argument(
        "--neighbors",
        action="store_true",
        help="Print graph-connected concepts",
    )
    show.add_argument(
        "--raw",
        action="store_true",
        help="Print the concept markdown file as-is",
    )
    show.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    # --- open ---
    op = subparsers.add_parser(
        "open",
        help="Open a concept file (or the graph JSON) in $EDITOR",
    )
    op.add_argument("slug", nargs="?", default=None)
    op.add_argument(
        "--graph",
        action="store_true",
        help="Open the knowledge graph JSON",
    )
    op.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
    )

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline."""
    config = MindForgeConfig(
        transcripts_dir=args.input,
        output_dir=args.output,
        use_embeddings=args.embeddings,
        similarity_threshold=args.similarity_threshold,
        use_llm=args.llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
    )

    print("MindForge v0.1.0")
    print(f"Input:  {config.transcripts_dir.resolve()}")
    print(f"Output: {config.output_dir.resolve()}")
    if config.use_llm:
        print(f"LLM:    {config.llm_provider}/{config.llm_model}")
    print()

    pipeline = MindForgePipeline(config)
    result = pipeline.run(dry_run=args.dry_run)

    print()
    if result.dry_run:
        print("Dry run — no files written.")
        print(f"  Would keep {result.concepts_after_dedup} concepts.")
        print(
            f"  New: {result.new}  Updated: {result.updated}  "
            f"Unchanged: {result.unchanged}  Removed: {result.removed}"
        )
        print("  Run without --dry-run to apply.")
    else:
        print(result.summary())
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the knowledge base, optionally filtered by tag/confidence/date."""
    from mindforge.query.engine import filter_concepts

    config = MindForgeConfig(
        output_dir=args.output,
        use_embeddings=args.embeddings,
    )

    pipeline = MindForgePipeline(config)
    pipeline._load_state()
    if pipeline.query_engine is None:
        print("No knowledge base found. Run 'mindforge ingest' first.", file=sys.stderr)
        return 1

    results = pipeline.query_engine.search(args.question, top_k=args.top_k)

    # Apply filters post-search so semantic scoring isn't distorted.
    if args.tag or args.min_confidence is not None or args.since:
        kept_concepts = filter_concepts(
            [r.concept for r in results],
            tag=args.tag,
            min_confidence=args.min_confidence,
            since=args.since,
        )
        kept_slugs = {c.slug for c in kept_concepts}
        results = [r for r in results if r.concept.slug in kept_slugs]

    print(pipeline.query_engine.format_results(results))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List concepts with optional filters."""
    from mindforge.distillation.concept import ConceptStore
    from mindforge.query.engine import filter_concepts

    config = MindForgeConfig(output_dir=args.output)
    manifest = config.output_dir / "concepts.json"
    if not manifest.exists():
        print("No knowledge base found. Run 'mindforge ingest' first.", file=sys.stderr)
        return 1

    store = ConceptStore.load(manifest)
    results = filter_concepts(
        store.all(),
        tag=args.tag,
        min_confidence=args.min_confidence,
        since=args.since,
    )
    if args.limit:
        results = results[: args.limit]
    for c in results:
        print(f"[{c.confidence:.2f}] {c.slug}  —  {c.name}")
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

    print("MindForge Knowledge Base Statistics")
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
        print("  Graph:")
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

    # Review queue summary (Phase 1.3).
    from collections import Counter

    from mindforge.hygiene.review_queue import build_review_queue

    queue = build_review_queue(store, half_life_days=config.decay_half_life_days)
    if queue:
        counts = Counter(item["reason"] for item in queue)
        print()
        print("  Review queue:")
        print(f"    Conflicted:  {counts.get('conflicted', 0)}")
        print(f"    Stale:       {counts.get('stale', 0)}")
        print(f"    Orphaned:    {counts.get('orphaned', 0)}")

    return 0


def compute_diff(manifest_path: Path, since: str | None = None) -> dict[str, list[str]]:
    """Compare the latest manifest snapshot against the prior one.

    When ``since`` is provided, compare against the oldest snapshot with a
    timestamp >= ``since`` (other than the current one), falling back to the
    immediately-prior snapshot if none match.
    """
    from mindforge.pipeline import read_manifest_history

    history = read_manifest_history(manifest_path)
    if len(history) < 2:
        return {"added": [], "modified": [], "deleted": []}
    current = history[-1]
    prior: dict[str, Any]
    if since:
        candidates = [h for h in history if h["timestamp"] >= since and h is not current]
        prior = candidates[0] if candidates else history[-2]
    else:
        prior = history[-2]
    cur_hashes = current["slug_hashes"]
    prev_hashes = prior["slug_hashes"]
    added = sorted(set(cur_hashes) - set(prev_hashes))
    deleted = sorted(set(prev_hashes) - set(cur_hashes))
    modified = sorted(
        slug for slug in cur_hashes if slug in prev_hashes and cur_hashes[slug] != prev_hashes[slug]
    )
    return {"added": added, "modified": modified, "deleted": deleted}


def cmd_diff(args: argparse.Namespace) -> int:
    """Print added/modified/deleted concepts since the previous snapshot."""
    config = MindForgeConfig(output_dir=args.output)
    d = compute_diff(config.output_dir / "manifest.json", since=args.since)
    if not (d["added"] or d["modified"] or d["deleted"]):
        print("No changes since previous run.")
        return 0
    print(f"Added ({len(d['added'])}):")
    for s in d["added"]:
        print(f"  + {s}")
    print(f"Modified ({len(d['modified'])}):")
    for s in d["modified"]:
        print(f"  ~ {s}")
    print(f"Deleted ({len(d['deleted'])}):")
    for s in d["deleted"]:
        print(f"  - {s}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Walk the hygiene review queue."""
    from mindforge.distillation.concept import ConceptStore
    from mindforge.hygiene.tui import review_loop

    config = MindForgeConfig(output_dir=args.output)
    manifest = config.output_dir / "concepts.json"
    if not manifest.exists():
        print("No knowledge base found. Run 'mindforge ingest' first.", file=sys.stderr)
        return 1

    store = ConceptStore.load(manifest)
    review_loop(store, half_life_days=config.decay_half_life_days)
    store.save(manifest)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the evaluation harness and write a JSON report."""
    from datetime import datetime, timezone

    from mindforge.eval.runner import render_markdown, run_eval

    if not args.fixtures.is_dir():
        print(f"Fixtures directory not found: {args.fixtures}", file=sys.stderr)
        return 1

    report = run_eval(args.fixtures, mode=args.mode)
    print(render_markdown(report))

    args.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import json

    (args.reports / f"{stamp}.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return 0


def _render_show(
    concept: Concept,
    *,
    sources: bool,
    neighbors: bool,
    raw: bool,
    store: ConceptStore | None,
) -> str:
    """Format a concept for CLI display. Extracted for test visibility."""
    from mindforge.distillation.renderer import render_concept

    if raw:
        return render_concept(concept)

    lines: list[str] = []
    lines.append(concept.name)
    lines.append(f"  {concept.definition}")
    if sources:
        if concept.sources:
            lines.append("Sources:")
            for s in concept.sources:
                turns = ", ".join(str(i) for i in s.turn_indices)
                lines.append(f"  {s.transcript_path} (turns {turns})")
        else:
            lines.append("Sources: (none)")
    if neighbors:
        if concept.relationships:
            lines.append("Neighbors:")
            for r in concept.relationships:
                lines.append(f"  {r.rel_type.value} → {r.target}")
        else:
            lines.append("Neighbors: (none)")
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> int:
    """Show a single concept, optionally with sources, neighbors, or raw markdown."""
    from mindforge.distillation.concept import ConceptStore

    config = MindForgeConfig(output_dir=args.output)
    manifest = config.output_dir / "concepts.json"
    if not manifest.exists():
        print("No knowledge base found. Run 'mindforge ingest' first.", file=sys.stderr)
        return 1

    store = ConceptStore.load(manifest)
    concept = store.get(args.slug)
    if not concept:
        print(f"Unknown concept: {args.slug}", file=sys.stderr)
        return 1

    print(
        _render_show(
            concept,
            sources=args.sources,
            neighbors=args.neighbors,
            raw=args.raw,
            store=store,
        )
    )
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    """Open a concept file or the graph JSON in $EDITOR."""
    import os
    import subprocess

    editor = os.environ.get("EDITOR", "vi")
    config = MindForgeConfig(output_dir=args.output)

    if args.graph:
        target = config.graph_dir / "knowledge_graph.json"
        if not target.exists():
            print("No graph yet. Run 'mindforge ingest' first.", file=sys.stderr)
            return 1
        return subprocess.call([editor, str(target)])

    if not args.slug:
        print("Provide a slug or use --graph.", file=sys.stderr)
        return 1
    target = config.concepts_dir / f"{args.slug}.md"
    if not target.exists():
        print(f"Concept file not found: {target}", file=sys.stderr)
        return 1
    return subprocess.call([editor, str(target)])


def cmd_mcp(args: argparse.Namespace) -> int:
    """Start the multi-KB MCP server.

    The server reads MINDFORGE_ROOT from the environment (defaults to
    ~/.mindforge). The legacy --output flag is kept for backwards
    compatibility but no longer gates server startup.
    """
    from mindforge.mcp.server import create_server

    config = MindForgeConfig(output_dir=args.output)
    server = create_server(config)
    server.run()
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
        "mcp": cmd_mcp,
        "show": cmd_show,
        "eval": cmd_eval,
        "review": cmd_review,
        "diff": cmd_diff,
        "list": cmd_list,
        "open": cmd_open,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
