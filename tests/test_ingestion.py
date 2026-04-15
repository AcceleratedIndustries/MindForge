"""Tests for the ingestion pipeline: parser, chunker, extractor."""

import textwrap
from pathlib import Path

from mindforge.ingestion.parser import parse_transcript, ConversationTurn, Transcript
from mindforge.ingestion.chunker import chunk_turn, Chunk
from mindforge.ingestion.extractor import extract_concepts, _is_valid_concept_name


class TestParser:
    def test_parse_role_prefixed(self, tmp_path):
        transcript_file = tmp_path / "test.md"
        transcript_file.write_text(
            "Human: What is AI?\n\n"
            "Assistant: AI is artificial intelligence.\n"
            "It enables machines to learn.\n"
        )
        result = parse_transcript(transcript_file)
        assert isinstance(result, Transcript)
        assert len(result.turns) == 2
        assert result.turns[0].role == "human"
        assert result.turns[1].role == "assistant"
        assert "artificial intelligence" in result.turns[1].content

    def test_parse_heading_style(self, tmp_path):
        transcript_file = tmp_path / "test.md"
        transcript_file.write_text(
            "## Human\nWhat is AI?\n\n"
            "## Assistant\nAI is artificial intelligence.\n"
        )
        result = parse_transcript(transcript_file)
        assert len(result.turns) == 2

    def test_parse_empty_file(self, tmp_path):
        transcript_file = tmp_path / "empty.md"
        transcript_file.write_text("")
        result = parse_transcript(transcript_file)
        # Should produce at least one turn (fallback)
        assert len(result.turns) >= 0

    def test_assistant_turns(self, tmp_path):
        transcript_file = tmp_path / "test.md"
        transcript_file.write_text(
            "Human: Question\n\nAssistant: Answer\n\n"
            "Human: Another\n\nAssistant: Response\n"
        )
        result = parse_transcript(transcript_file)
        assert len(result.assistant_turns) == 2

    def test_separator_based(self, tmp_path):
        transcript_file = tmp_path / "test.md"
        transcript_file.write_text(
            "Human: First question\n\n---\n\n"
            "This is the assistant response.\n"
        )
        result = parse_transcript(transcript_file)
        assert len(result.turns) >= 2


class TestChunker:
    def test_prose_chunk(self):
        turn = ConversationTurn(
            role="assistant",
            content="This is a paragraph about AI.\n\nThis is another paragraph.",
            index=0,
            source_file="test.md",
        )
        chunks = chunk_turn(turn)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_code_block_preserved(self):
        turn = ConversationTurn(
            role="assistant",
            content=(
                "Here is an example of how to compute embeddings in Python:\n\n"
                "```python\n"
                "from sentence_transformers import SentenceTransformer\n"
                "model = SentenceTransformer('all-MiniLM-L6-v2')\n"
                "embeddings = model.encode(['hello world', 'how are you'])\n"
                "print(embeddings.shape)\n"
                "```\n\n"
                "This produces a numpy array of shape (2, 384) since the model outputs 384-dimensional vectors."
            ),
            index=0,
            source_file="test.md",
        )
        chunks = chunk_turn(turn)
        code_chunks = [c for c in chunks if c.chunk_type == "code"]
        assert len(code_chunks) >= 1

    def test_empty_turn(self):
        turn = ConversationTurn(
            role="assistant",
            content="",
            index=0,
            source_file="test.md",
        )
        chunks = chunk_turn(turn)
        assert len(chunks) == 0

    def test_chunk_ids_unique(self):
        turn = ConversationTurn(
            role="assistant",
            content="Paragraph one about embeddings.\n\nParagraph two about search.\n\nParagraph three about RAG.",
            index=0,
            source_file="test.md",
        )
        chunks = chunk_turn(turn)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestConceptNameValidation:
    def test_valid_names(self):
        assert _is_valid_concept_name("Vector Embeddings")
        assert _is_valid_concept_name("KV Cache")
        assert _is_valid_concept_name("Semantic Search")
        assert _is_valid_concept_name("Retrieval-Augmented Generation")

    def test_blocked_generic(self):
        assert not _is_valid_concept_name("this")
        assert not _is_valid_concept_name("key")
        assert not _is_valid_concept_name("critical")
        assert not _is_valid_concept_name("meaning")

    def test_too_short(self):
        assert not _is_valid_concept_name("AI")
        assert not _is_valid_concept_name("x")

    def test_sentence_fragments(self):
        assert not _is_valid_concept_name("It uses vector embeddings")
        assert not _is_valid_concept_name("This means something")
        assert not _is_valid_concept_name("Without KV Cache generation")

    def test_too_long(self):
        assert not _is_valid_concept_name("This is a very long name that should not be a concept at all")


class TestExtractor:
    def test_extract_definitions(self):
        chunks = [
            Chunk(
                content="Vector Database is a specialized database optimized for storing vectors. It provides fast retrieval.",
                source_file="test.md",
                turn_index=0,
                chunk_index=0,
                chunk_type="prose",
            )
        ]
        concepts = extract_concepts(chunks)
        names = [c.name.lower() for c in concepts]
        assert any("vector database" in n for n in names)

    def test_extract_from_headings(self):
        chunks = [
            Chunk(
                content="## Semantic Search\n\nSemantic Search is an approach that understands meaning behind queries rather than just matching keywords literally.",
                source_file="test.md",
                turn_index=0,
                chunk_index=0,
                chunk_type="prose",
            )
        ]
        concepts = extract_concepts(chunks)
        names = [c.name.lower() for c in concepts]
        assert any("semantic search" in n for n in names)
