# Choosing a Database for the Ingestion Pipeline

---

We need persistent storage for the knowledge base. Three candidates: Postgres, DuckDB, and SQLite.

---

**Postgres** is a mature relational database with rich extensions. It has pgvector for semantic search and is the default choice for production web apps. The downside is operational overhead — you need to run a server, manage users, tune config.

---

**DuckDB** is an embedded analytical database that runs in-process. It is faster than Postgres for analytic queries and has zero operational cost. The downside is it is not designed for high-concurrency writes.

---

**SQLite** is the classic embedded database. It is battle-tested and universally available. The downside is single-writer semantics and limited full-text search.

---

For a single-user local tool where reads dominate and writes are batched, DuckDB is the right choice. We do not need the concurrency story that Postgres offers, and SQLite is slower for analytical workloads.
