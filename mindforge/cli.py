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

from mindforge import __version__
from mindforge.config import MindForgeConfig
from mindforge.config_file import ConfigFile, load_config, merge_with_overrides
from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.embeddings.index import Embedder
from mindforge.pipeline import MindForgePipeline
from mindforge.query.engine import RetrievalWeights


def _build_embedder(file_cfg: ConfigFile) -> Embedder | None:
    """Construct an embedding provider from the merged config. None means use
    the default sentence-transformers path inside EmbeddingIndex."""
    provider = file_cfg.embeddings.provider
    if provider == "ollama":
        from mindforge.embeddings.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=file_cfg.embeddings.base_url or "http://localhost:11434",
            model=file_cfg.embeddings.model or "nomic-embed-text",
        )
    if provider == "openai-compat":
        from mindforge.embeddings.openai_compat_provider import (
            OpenAICompatibleEmbeddingProvider,
        )

        return OpenAICompatibleEmbeddingProvider(
            base_url=file_cfg.embeddings.base_url or "http://localhost:8080/v1",
            model=file_cfg.embeddings.model or "text-embedding-3-small",
            api_key=file_cfg.embeddings.api_key,
        )
    return None


def _load_merged_config(args: argparse.Namespace) -> ConfigFile:
    """Load the YAML config file and apply CLI overrides. None/empty CLI flags
    are skipped so the file (or dataclass defaults) survive."""
    file_cfg = load_config()
    return merge_with_overrides(
        file_cfg,
        llm_provider=getattr(args, "llm_provider", None),
        llm_base_url=getattr(args, "llm_base_url", None),
        llm_model=getattr(args, "llm_model", None),
        llm_api_key=getattr(args, "llm_api_key", None),
        llm_summarize_model=getattr(args, "llm_summarize_model", None),
        llm_timeout=getattr(args, "llm_timeout", None),
        embeddings_provider=getattr(args, "embedding_provider", None),
        embeddings_base_url=getattr(args, "embedding_base_url", None),
        embeddings_model=getattr(args, "embedding_model", None),
        embeddings_api_key=getattr(args, "embedding_api_key", None),
    )


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
    ingest.add_argument(
        "--full",
        action="store_true",
        help="Force a full rebuild, ignoring the incremental hash cache",
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
        default=None,
        help="LLM provider (overrides config; default: from config or ollama)",
    )
    llm_group.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (overrides config; default: from config or qwen3:30b-a3b)",
    )
    llm_group.add_argument(
        "--llm-base-url",
        default="",
        help="LLM API base URL (overrides config; default: auto-detect from provider)",
    )
    llm_group.add_argument(
        "--llm-api-key",
        default="",
        help="API key for OpenAI-compatible providers (overrides config)",
    )

    # Embedding provider options
    emb_group = ingest.add_argument_group("Embedding provider")
    emb_group.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "ollama", "openai-compat"],
        default=None,
        help="Embedding provider (overrides config; default: from config or sentence-transformers)",
    )
    emb_group.add_argument(
        "--embedding-base-url",
        default="",
        help="Base URL for ollama or openai-compat providers",
    )
    emb_group.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name (provider-specific default if unset)",
    )
    emb_group.add_argument(
        "--embedding-api-key",
        default="",
        help="API key for openai-compat provider",
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
    query.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "semantic"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    query.add_argument(
        "--weights",
        type=str,
        default=None,
        metavar="K,S,G",
        help=(
            "Override default weights as comma-separated keyword,semantic,graph (e.g. 0.4,0.4,0.2)"
        ),
    )
    query.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include soft-deleted concepts in results",
    )
    qemb_group = query.add_argument_group("Embedding provider")
    qemb_group.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "ollama", "openai-compat"],
        default=None,
        help="Embedding provider (overrides config; default: from config or sentence-transformers)",
    )
    qemb_group.add_argument(
        "--embedding-base-url",
        default="",
        help="Base URL for ollama or openai-compat providers",
    )
    qemb_group.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name (provider-specific default if unset)",
    )
    qemb_group.add_argument(
        "--embedding-api-key",
        default="",
        help="API key for openai-compat provider",
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
        "--include-deleted",
        action="store_true",
        help="Include soft-deleted concepts in results",
    )
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
    mcp_llm = mcp.add_argument_group("LLM (synthesis tools)")
    mcp_llm.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM provider (overrides config)",
    )
    mcp_llm.add_argument("--llm-base-url", default="", help="LLM base URL (overrides config)")
    mcp_llm.add_argument("--llm-model", default=None, help="LLM model name (overrides config)")
    mcp_llm.add_argument(
        "--llm-summarize-model",
        default="",
        help="Optional dedicated model for summarize_query (overrides config)",
    )
    mcp_llm.add_argument(
        "--llm-api-key",
        default="",
        help="API key for OpenAI-compatible providers (overrides config)",
    )
    mcp_llm.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="LLM request timeout in seconds (overrides config)",
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
        choices=["heuristic", "llm", "tune-retrieval"],
        default="heuristic",
        help="Extraction mode, or tune-retrieval to sweep hybrid weights",
    )
    ev.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="(tune-retrieval only) pre-ingested KB to sweep weights against",
    )
    ev.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="(tune-retrieval only) top-k for recall@k metric (default: 5)",
    )
    ev.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="(tune-retrieval only) weight grid step size (default: 0.1)",
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
        "--include-deleted",
        action="store_true",
        help="Include soft-deleted concepts in results",
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

    # --- config ---
    config_p = subparsers.add_parser(
        "config",
        help="Show or initialize the MindForge config file",
    )
    config_sub = config_p.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="Print the merged effective config")
    config_init = config_sub.add_parser(
        "init",
        help="Write a commented config template to ~/.config/mindforge/config.yaml",
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config file",
    )

    # --- prune ---
    prune_p = subparsers.add_parser(
        "prune",
        help="Hard-delete soft-marked concepts and their on-disk artifacts",
    )
    prune_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without changing anything",
    )
    prune_p.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        metavar="N",
        help="Only prune concepts soft-deleted at least N days ago",
    )
    prune_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline."""
    file_cfg = _load_merged_config(args)
    config = MindForgeConfig(
        transcripts_dir=args.input,
        output_dir=args.output,
        use_embeddings=args.embeddings,
        similarity_threshold=args.similarity_threshold,
        use_llm=args.llm,
        llm_provider=file_cfg.llm.provider,
        llm_model=file_cfg.llm.model,
        llm_base_url=file_cfg.llm.base_url,
        llm_api_key=file_cfg.llm.api_key,
        llm_keep_alive=file_cfg.llm.keep_alive,
        llm_timeout=file_cfg.llm.timeout,
        embedding_provider=_build_embedder(file_cfg) if args.embeddings else None,
    )

    print(f"MindForge v{__version__}")
    print(f"Input:  {config.transcripts_dir.resolve()}")
    print(f"Output: {config.output_dir.resolve()}")
    if config.use_llm:
        print(f"LLM:    {config.llm_provider}/{config.llm_model}")
    print()

    pipeline = MindForgePipeline(config)
    if args.full:
        pipeline._force_full = True
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


def _parse_weights(raw: str | None) -> RetrievalWeights | None:
    """Parse a ``--weights K,S,G`` CLI string into a RetrievalWeights instance.

    Returns None if ``raw`` is falsy. Calls ``sys.exit`` with a clear message
    when the input is not three comma-separated floats.
    """
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 3:
        raise SystemExit("--weights must be three comma-separated floats: keyword,semantic,graph")
    try:
        floats = [float(x) for x in parts]
    except ValueError as exc:
        raise SystemExit(
            "--weights must be three comma-separated floats: keyword,semantic,graph"
        ) from exc
    return RetrievalWeights(keyword=floats[0], semantic=floats[1], graph=floats[2])


def cmd_query(args: argparse.Namespace) -> int:
    """Query the knowledge base, optionally filtered by tag/confidence/date."""
    from mindforge.query.engine import filter_concepts

    file_cfg = _load_merged_config(args)
    config = MindForgeConfig(
        output_dir=args.output,
        use_embeddings=args.embeddings,
        embedding_provider=_build_embedder(file_cfg) if args.embeddings else None,
    )

    pipeline = MindForgePipeline(config)
    pipeline._load_state()
    if pipeline.query_engine is None:
        print("No knowledge base found. Run 'mindforge ingest' first.", file=sys.stderr)
        return 1

    weights = _parse_weights(getattr(args, "weights", None))
    if weights is None:
        fw = file_cfg.retrieval.weights
        weights = RetrievalWeights(
            keyword=fw.get("keyword", 0.4),
            semantic=fw.get("semantic", 0.4),
            graph=fw.get("graph", 0.2),
        )
    mode = getattr(args, "mode", "hybrid")
    results = pipeline.query_engine.search(
        args.question, top_k=args.top_k, mode=mode, weights=weights
    )

    # Apply filters post-search so semantic scoring isn't distorted.
    include_deleted = args.include_deleted
    kept_concepts = filter_concepts(
        [r.concept for r in results],
        tag=args.tag,
        min_confidence=args.min_confidence,
        since=args.since,
        include_deleted=include_deleted,
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
        include_deleted=args.include_deleted,
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
    import json
    from datetime import datetime, timezone

    from mindforge.eval.runner import (
        render_markdown,
        render_tune_markdown,
        run_eval,
        run_tune_retrieval,
    )

    args.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.mode == "tune-retrieval":
        if not args.output.is_dir():
            print(f"Output directory not found: {args.output}", file=sys.stderr)
            return 1
        report = run_tune_retrieval(args.output, k=args.top_k, step=args.step)
        print(render_tune_markdown(report))
        (args.reports / f"{stamp}-tune.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return 0

    if not args.fixtures.is_dir():
        print(f"Fixtures directory not found: {args.fixtures}", file=sys.stderr)
        return 1

    report = run_eval(args.fixtures, mode=args.mode)
    print(render_markdown(report))

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
    if not args.include_deleted and concept.status == "deleted":
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
    import subprocess  # nosec B404 — launching $EDITOR; shell=False is intentional

    editor = os.environ.get("EDITOR", "vi")
    config = MindForgeConfig(output_dir=args.output)

    if args.graph:
        target = config.graph_dir / "knowledge_graph.json"
        if not target.exists():
            print("No graph yet. Run 'mindforge ingest' first.", file=sys.stderr)
            return 1
        # $EDITOR is user-controlled on purpose; no shell involved.
        return subprocess.call([editor, str(target)])  # nosec B603

    if not args.slug:
        print("Provide a slug or use --graph.", file=sys.stderr)
        return 1
    target = config.concepts_dir / f"{args.slug}.md"
    if not target.exists():
        print(f"Concept file not found: {target}", file=sys.stderr)
        return 1
    # $EDITOR is user-controlled on purpose; no shell involved.
    return subprocess.call([editor, str(target)])  # nosec B603


def cmd_mcp(args: argparse.Namespace) -> int:
    """Start the multi-KB MCP server.

    The server reads MINDFORGE_ROOT from the environment (defaults to
    ~/.mindforge). The legacy --output flag is kept for backwards
    compatibility but no longer gates server startup.
    """
    from mindforge.mcp.server import create_server

    config = MindForgeConfig(output_dir=args.output)
    file_cfg = _load_merged_config(args)
    server = create_server(config, file_cfg=file_cfg)
    server.run()
    return 0


_CONFIG_TEMPLATE = """# MindForge config
# See docs/superpowers/specs/2026-05-05-phase3-v0.3.0-design.md for the spec.

llm:
  provider: ollama          # ollama | openai
  base_url: http://localhost:11434
  model: qwen3:30b-a3b
  keep_alive: -1
  timeout: 120
  api_key: ""
  # Synthesis tools (summarize_query, explain_concept, compare_concepts,
  # path_between) ride this knob. True = let reasoning models (qwen3,
  # deepseek-r1, gpt-oss, ...) think before answering. Null = server default.
  # Extraction always disables thinking; deliberation is wasted on JSON-shape
  # output and runs on every chunk.
  # synthesis_think: true
  # Optional: route summarize_query to a bigger model. Useful for metered APIs.
  # summarize_model: nemotron-3-super:latest

embeddings:
  provider: sentence-transformers   # sentence-transformers | ollama | openai-compat
  base_url: ""
  model: ""
  api_key: ""

retrieval:
  weights:
    keyword: 0.4
    semantic: 0.4
    graph: 0.2
  seed_pool_size: 10
  walk_depth: 2
"""


def cmd_config(args: argparse.Namespace) -> int:
    import yaml

    from mindforge.config_file import default_config_path, load_config

    if args.config_command == "show":
        p = default_config_path()
        cfg = load_config(p)
        source = str(p) if p.exists() else "(defaults — no config file found)"
        print(f"# Loaded from: {source}\n")
        out = {
            "llm": cfg.llm.__dict__,
            "embeddings": cfg.embeddings.__dict__,
            "retrieval": cfg.retrieval.__dict__,
        }
        print(yaml.safe_dump(out, sort_keys=False), end="")
        return 0

    if args.config_command == "init":
        p = default_config_path()
        if p.exists() and not args.force:
            print(
                f"Config file already exists at {p}. Use --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_CONFIG_TEMPLATE)
        print(f"Wrote template config to {p}")
        return 0

    print("Run 'mindforge config show' or 'mindforge config init'.", file=sys.stderr)
    return 1


def cmd_prune(args: argparse.Namespace) -> int:
    """Hard-delete soft-marked concepts."""
    from mindforge.prune import prune_orphans

    config = MindForgeConfig(output_dir=args.output)
    summary = prune_orphans(config, dry_run=args.dry_run, older_than_days=args.older_than_days)

    if args.dry_run:
        print(f"Would remove {summary.would_remove} soft-deleted concept(s):")
        for slug in summary.slugs:
            print(f"  - {slug}")
        print("Re-run without --dry-run to apply.")
    else:
        print(f"Removed {summary.removed} soft-deleted concept(s).")
        for slug in summary.slugs:
            print(f"  - {slug}")
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
        "config": cmd_config,
        "prune": cmd_prune,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
