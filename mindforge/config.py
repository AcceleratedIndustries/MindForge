"""Configuration for the MindForge pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mindforge.paths import MindForgePaths


@dataclass
class MindForgeConfig:
    """Central configuration for the MindForge pipeline."""

    # Input
    transcripts_dir: Path = Path("examples/transcripts")

    # Output
    output_dir: Path = Path("output")

    # Concept extraction
    min_concept_length: int = 20
    max_concept_length: int = 5000
    similarity_threshold: float = 0.75  # For deduplication

    # Linking
    link_confidence_threshold: float = 0.3

    # LLM extraction (optional)
    use_llm: bool = False
    llm_provider: str = "ollama"  # "ollama" or "openai"
    llm_model: str = "llama3.2"
    llm_base_url: str = ""  # Auto-set based on provider if empty
    llm_api_key: str = ""  # Required for OpenAI provider

    # Embeddings (optional)
    embedding_model: str = "all-MiniLM-L6-v2"
    use_embeddings: bool = False

    # Multi-KB root (env/config precedence via MindForgePaths).
    # None → resolve lazily in __post_init__.
    kb_root: Optional[Path] = None

    # Hygiene (Phase 1.3)
    decay_half_life_days: float = 62.0

    # Derived paths
    concepts_dir: Path = field(init=False)
    graph_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    provenance_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.kb_root is None:
            self.kb_root = MindForgePaths.resolve().root
        self.concepts_dir = self.output_dir / "concepts"
        self.graph_dir = self.output_dir / "graph"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.provenance_dir = self.output_dir / "provenance"

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in [self.concepts_dir, self.graph_dir, self.embeddings_dir, self.provenance_dir]:
            d.mkdir(parents=True, exist_ok=True)
