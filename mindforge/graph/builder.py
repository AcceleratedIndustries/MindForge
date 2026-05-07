"""Graph builder: constructs a knowledge graph from concepts and their relationships.

Uses NetworkX for in-memory graph operations and exports to JSON
for portability and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from mindforge.distillation.concept import Concept, ConceptStore


class KnowledgeGraph:
    """A knowledge graph built from MindForge concepts.

    Nodes represent concepts, edges represent typed relationships.
    Supports export to JSON and (when networkx is available) graph analysis.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}  # slug -> node data
        self._edges: list[dict] = []

        if HAS_NETWORKX:
            self._graph: nx.DiGraph | None = nx.DiGraph()
        else:
            self._graph = None

    def add_concept(self, concept: Concept) -> None:
        """Add a concept as a node in the graph."""
        node_data = {
            "id": concept.slug,
            "label": concept.name,
            "definition": concept.definition[:200],
            "tags": concept.tags,
            "confidence": concept.confidence,
        }
        self._nodes[concept.slug] = node_data

        if self._graph is not None:
            self._graph.add_node(concept.slug, **node_data)

    def add_relationships(self, concept: Concept) -> None:
        """Add all relationships from a concept as edges."""
        for rel in concept.relationships:
            if rel.target not in self._nodes:
                continue
            edge_data = {
                "source": rel.source,
                "target": rel.target,
                "type": rel.rel_type.value,
                "confidence": rel.confidence,
            }
            self._edges.append(edge_data)

            if self._graph is not None:
                self._graph.add_edge(
                    rel.source,
                    rel.target,
                    type=rel.rel_type.value,
                    confidence=rel.confidence,
                )

    @classmethod
    def from_store(cls, store: ConceptStore) -> KnowledgeGraph:
        """Build a graph from a ConceptStore. Deleted concepts are excluded."""
        graph = cls()
        active = [c for c in store.all() if c.status != "deleted"]
        for concept in active:
            graph.add_concept(concept)
        for concept in active:
            graph.add_relationships(concept)
        return graph

    def to_json(self) -> dict:
        """Export the graph as a JSON-serializable dictionary."""
        return {
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
            "metadata": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
        }

    def save(self, path: Path) -> None:
        """Save the graph to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        """Load a graph from a JSON file."""
        graph = cls()
        data = json.loads(path.read_text())
        for node in data.get("nodes", []):
            graph._nodes[node["id"]] = node
            if graph._graph is not None:
                graph._graph.add_node(node["id"], **node)
        for edge in data.get("edges", []):
            graph._edges.append(edge)
            if graph._graph is not None:
                graph._graph.add_edge(
                    edge["source"],
                    edge["target"],
                    type=edge["type"],
                    confidence=edge.get("confidence", 1.0),
                )
        return graph

    # --- Analysis methods (require NetworkX) ---

    def neighbors(self, slug: str) -> list[str]:
        """Get all directly connected concept slugs."""
        if self._graph is not None:
            preds = list(self._graph.predecessors(slug))
            succs = list(self._graph.successors(slug))
            return list(dict.fromkeys(preds + succs))
        # Fallback without networkx
        connected = set()
        for edge in self._edges:
            if edge["source"] == slug:
                connected.add(edge["target"])
            elif edge["target"] == slug:
                connected.add(edge["source"])
        return list(connected)

    def subgraph(
        self,
        center: str,
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> dict:
        """Return a JSON-shape subgraph centered on ``center``, depth-limited.

        The result has the form ``{"nodes": [{"id", "label"}, ...],
        "edges": [{"source", "target", "type", "confidence"}, ...]}``.
        Edges are kept directed; ``depth`` is measured against an undirected
        projection so reachable concepts are included regardless of direction.
        """
        if self._graph is None or center not in self._graph:
            return {"nodes": [], "edges": []}
        nodes_in_radius = nx.single_source_shortest_path_length(
            self._graph.to_undirected(),
            center,
            cutoff=depth,
        )
        sub = self._graph.subgraph(nodes_in_radius.keys())
        edges: list[dict] = []
        for u, v, data in sub.edges(data=True):
            rel_type = data.get("type", "related_to")
            if edge_types and rel_type not in edge_types:
                continue
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "type": rel_type,
                    "confidence": float(data.get("confidence", 1.0)),
                }
            )
        nodes = [{"id": n, "label": sub.nodes[n].get("label", n)} for n in sub.nodes]
        return {"nodes": nodes, "edges": edges}

    def shortest_paths(
        self,
        source: str,
        target: str,
        max_length: int = 4,
        max_paths: int = 3,
    ) -> list[list[str]]:
        """Return up to ``max_paths`` shortest paths between ``source`` and
        ``target``, each containing at most ``max_length`` edges. Treats the
        graph as undirected so semantically-related concepts are reachable
        regardless of edge direction.
        """
        if self._graph is None or source not in self._graph or target not in self._graph:
            return []
        paths: list[list[str]] = []
        try:
            for p in nx.all_shortest_paths(self._graph.to_undirected(), source, target):
                if len(p) - 1 > max_length:
                    continue
                paths.append(list(p))
                if len(paths) >= max_paths:
                    break
        except nx.NetworkXNoPath:
            return []
        return paths

    def neighbors_with_confidence(self, slug: str) -> list[tuple[str, float]]:
        """Return [(neighbor_slug, edge_confidence)] for outgoing edges from slug."""
        if self._graph is not None:
            if slug not in self._graph:
                return []
            out: list[tuple[str, float]] = []
            for neighbor in self._graph.successors(slug):
                edge = self._graph[slug][neighbor]
                out.append((neighbor, float(edge.get("confidence", 1.0))))
            return out
        out_fb: list[tuple[str, float]] = []
        for edge in self._edges:
            if edge["source"] == slug:
                out_fb.append((edge["target"], float(edge.get("confidence", 1.0))))
        return out_fb

    def nodes(self) -> list[str]:
        """Return all node slugs."""
        if self._graph is not None:
            return list(self._graph.nodes())
        return list(self._nodes.keys())

    def central_concepts(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the most central concepts by degree centrality."""
        if self._graph is not None and len(self._graph) > 0:
            centrality = nx.degree_centrality(self._graph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_n]

        # Fallback: count edge participation
        counts: dict[str, int] = {}
        for edge in self._edges:
            counts[edge["source"]] = counts.get(edge["source"], 0) + 1
            counts[edge["target"]] = counts.get(edge["target"], 0) + 1
        total = max(len(self._nodes), 1)
        sorted_nodes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(slug, count / total) for slug, count in sorted_nodes[:top_n]]

    def find_clusters(self) -> list[set[str]]:
        """Find clusters of closely related concepts."""
        if self._graph is not None and len(self._graph) > 0:
            undirected = self._graph.to_undirected()
            return [set(c) for c in nx.connected_components(undirected)]
        return [set(self._nodes.keys())]

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        result = {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "clusters": len(self.find_clusters()),
        }
        if self._graph is not None and len(self._graph) > 0:
            result["density"] = round(nx.density(self._graph), 4)
        return result
