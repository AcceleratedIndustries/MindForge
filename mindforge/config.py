"""Configuration for the MindForge pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


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

    # Derived paths
    concepts_dir: Path = field(init=False)
    graph_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.concepts_dir = self.output_dir / "concepts"
        self.graph_dir = self.output_dir / "graph"
        self.embeddings_dir = self.output_dir / "embeddings"

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in [self.concepts_dir, self.graph_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)
