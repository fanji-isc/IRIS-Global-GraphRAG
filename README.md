# InterSystems IRIS GraphRAG
GraphRAG with InterSystems IRIS — A research and demo project that loads academic papers and their relationships into InterSystems IRIS, builds embeddings, and enables semantic + graph-based retrieval using vector search and knowledge graph queries.
## Introduction

Traditional RAG pulls relevant chunks with vector search, but it can miss structure—who’s related to what, and how. GraphRAG adds that structure by blending graph context (entities + relations) with dense retrieval, so answers can be more grounded and explainable. In this demo, we use 50 recent GraphRAG-related papers (title, abstract, authors, URL, date pulled from arXiv) and a Graph Transformer to automatically extract entities (nodes) and relations (edges) from the text; the graph is stored in IRIS globals, while the same papers are embedded for dense retrieval.

This demo uses InterSystems IRIS in two complementary ways:

**IRIS globals as a graph store**
Globals are IRIS’s persistent, sparse, multi-dimensional arrays—perfect for graphs. We keep paper in ^GraphContent and relations in ^GraphRelations, which makes it trivial to fetch nodes/edges for any subset of documents.

**IRIS SQL for vector retrieval**
Each paper is embedded with MiniLM-L6-v2 (384d), stored in IRIS as VECTOR, and indexed with HNSW for fast approximate nearest-neighbor search.

## Demo Overview
This app lets you run the same question through three pipelines and compare results side-by-side—LLM (no retrieval), RAG (vector retrieval over papers), and GraphRAG (RAG + a graph neighborhood from IRIS globals) with an interactive entity-relation visualization.

Here’s a simple query flow starting with the agent and showing the three paths:

```text
User Question
      │
      ▼
 ┌───────────────┐
 │  Agent Router │  (classifies intent & picks mode / params)
 └──────┬────────┘
        │
   ┌────┼───────────────┬─────────────────────┐
   │    │               │                     │
   │    ▼               ▼                     ▼
   │  LLM            RAG                GraphRAG
   │ (no retrieval)  (vector-only)      (vector + graph)
   │                 ┌───────────────┐  ┌──────────────────┐
   │                 │ Embed query   │  │ Embed query      │
   │                 │ MiniLM-L6-v2  │  │ MiniLM-L6-v2     │
   │                 └───────┬───────┘  └─────────┬────────┘
   │                         ▼                    ▼
   │                 ANN search (HNSW)      ANN search (HNSW)
   │                 over IRIS SQL          over IRIS SQL
   │                 table `paper_content`  table `paper_content`
   │                         │                    │
   │                         ▼                    ▼
   │                  Top-K papers           Doc IDs → fetch
   │                                          nodes/edges from
   │                                          IRIS globals:
   │                                          ^GraphContent,
   │                                          ^GraphRelations
   │                         │                    │
   │                         ▼                    ▼
   │                  Generate answer        Build subgraph →
   │                                         Generate answer
   │                                         + graph visualization
   └──────────────────────┼──────────────────────────┼────────
                          ▼                          ▼
                     UI shows answer            UI shows answer
                                                + entity-relation
                                                visualization


```

## InterSystems IRIS GraphRAG Quickstart      
1. Clone the repo

    ```bash
          git clone https://github.com/fanji-isc/IRIS-Global-GraphRAG.git
    ```
2. Add your own OPEN AI key to .env file in app/

    ```bash

          OPENAI_API_KEY=sk-...
     ```
3. Start the Docker containers (one for IRIS, one for Flask):

     ```bash
          docker compose up --build
     ```
4. Open the app & IRIS
   
   Demo: http://localhost:8000

   IRIS Portal: http://localhost:52773
