# RAG Retrieval Pipeline Plan

A comprehensive plan for building a modern retrieval system based on course materials (notebooks 03-07) and updated with 2025-2026 best practices.

---

## 1. Overview: Course Material vs. Modern Approaches

### What the Course Covers

| Notebook | Technique | Libraries/Models Used |
|----------|-----------|----------------------|
| 03 | Semantic Chunking | `semantic-chunkers` (Aurelio Labs), `semantic-router` |
| 04 | Contextual Retrieval | Anthropic Claude Haiku + Prompt Caching |
| 05 | Reverse HyDE | OpenAI GPT-3.5 + `text-embedding-ada-002` |
| 06 | Hybrid Search | `bm25s` + Qdrant + `all-MiniLM-L6-v2` |
| 07 | Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### What's Outdated

| Issue | Course Approach | Modern Replacement |
|-------|-----------------|-------------------|
| Embedding Model | `text-embedding-ada-002`, `all-MiniLM-L6-v2` | `text-embedding-3-large`, `voyage-3`, `Cohere embed-v4`, `nomic-embed-text-v2` |
| Context Injection | LLM-generated context per chunk (expensive) | **Late Chunking** or **Contextualized Embedding Models** (`voyage-context-3`) |
| Hybrid Fusion | Weighted average normalization | **Reciprocal Rank Fusion (RRF)** or **Weighted RRF** |
| Reranker | Small cross-encoder (`ms-marco-MiniLM`) | **ColBERT v2**, `bge-reranker-v2`, `Cohere Rerank 3.5` |
| Query Handling | Single query → retrieval | **Multi-query expansion**, **Query decomposition** |

---

## 2. Recommended Modern Retrieval Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Documents → Semantic Chunking → Context Enrichment → Dual Indexing         │
│                  │                      │                   │               │
│                  ▼                      ▼                   ▼               │
│           Split by meaning      Late Chunking OR      Dense + Sparse        │
│           (256-512 tokens)      Contextual Embed      (Vector + BM25)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL PHASE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query → Query Understanding → Hybrid Search → Fusion → Reranking → Top-K   │
│            │                        │             │          │              │
│            ▼                        ▼             ▼          ▼              │
│     Multi-query OR           Dense + Sparse     RRF     ColBERT v2 OR       │
│     Decomposition            in parallel               Cross-encoder        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Plans

### 3.1 Chunking Strategy

**Recommendation: Semantic Chunking with Optimal Parameters**

The course's `StatisticalChunker` approach remains valid. Research confirms semantic chunking delivers ~70% accuracy improvement over naive methods.

**Modern Best Practices:**
- **Chunk size:** 256-512 tokens (optimal range from research)
- **Overlap:** 10-20% between chunks
- **Metadata:** Store `prechunk_id` and `postchunk_id` for context retrieval

**Implementation Options:**

```python
# Option A: Continue using semantic-chunkers (course approach)
from semantic_chunkers import StatisticalChunker

chunker = StatisticalChunker(
    encoder=encoder,
    min_split_tokens=100,  # Lower bound
    max_split_tokens=500,  # Upper bound
)

# Option B: LangChain's SemanticChunker (more maintained)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

# Option C: Simple recursive with overlap (for Turkish text)
from langchain.text_splitter import RecursiveCharacterTextSplitter

chunker = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**For SevanBot (Turkish Content):** Start with recursive splitting at paragraph boundaries, then test semantic chunking. Turkish-specific considerations:
- Ensure embedding model supports Turkish well (OpenAI, Cohere, or multilingual models)
- Test chunk boundaries with actual articles

---

### 3.2 Context Enrichment

**The Problem:** Individual chunks lose document-level context, hurting retrieval.

**Course Approach (04_contextual_retrieval.ipynb):**
- Use Claude Haiku to generate context for each chunk
- Prepend context to chunk before embedding
- Uses prompt caching to reduce costs

**Modern Alternatives:**

| Approach | Pros | Cons | Cost |
|----------|------|------|------|
| **LLM Contextual (Course)** | High quality context | Expensive, slow indexing | ~$0.25/1M tokens (cached) |
| **Late Chunking** | No LLM needed, uses long-context embedders | Requires compatible model | Just embedding cost |
| **Contextualized Embeddings** (`voyage-context-3`) | Drop-in replacement, best quality | API dependency | $0.06/1M tokens |
| **Document Title Prepending** | Simple, free | Limited context | Free |

**Recommended Approach for SevanBot:**

```python
# Simple but effective: Prepend document metadata
def create_chunk_with_context(chunk: str, doc_metadata: dict) -> str:
    """Add document-level context to chunk."""
    context_parts = []

    if doc_metadata.get("title"):
        context_parts.append(f"Article: {doc_metadata['title']}")
    if doc_metadata.get("date"):
        context_parts.append(f"Date: {doc_metadata['date']}")
    if doc_metadata.get("subtitle"):
        context_parts.append(f"Summary: {doc_metadata['subtitle']}")

    context = " | ".join(context_parts)
    return f"{context}\n\n{chunk}"
```

**Advanced Option: Late Chunking (if using Jina or similar)**

```python
# Late chunking preserves context without LLM
# Requires long-context embedding model
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Embed full document first, then chunk embeddings
doc_embeddings = model.encode_documents([full_document])
chunk_embeddings = model.extract_chunk_embeddings(doc_embeddings, chunk_boundaries)
```

---

### 3.3 Reverse HyDE (Hypothetical Questions)

**Course Approach (05_reverse_hyde.ipynb):**
- Generate 3-5 hypothetical questions per chunk
- Index questions alongside chunks
- Match user queries to questions (better semantic match)

**Modern Assessment:**

This technique is still valuable but consider trade-offs:
- **Pros:** Bridges query-document vocabulary gap, improves recall
- **Cons:** 3-5x storage per chunk, LLM cost at indexing time

**When to Use:**
- FAQs, knowledge bases where users ask questions
- Technical documentation
- NOT recommended for narrative content (like essays/articles)

**For SevanBot:** Consider skipping Reverse HyDE initially. The articles are essays/opinion pieces, not Q&A content. The query-document mismatch may not be significant.

---

### 3.4 Embedding Model Selection

**Course Uses:** `text-embedding-ada-002` (deprecated), `all-MiniLM-L6-v2` (small)

**2025-2026 Recommendations:**

| Model | MTEB Score | Dimensions | Multilingual | Cost/1M tokens |
|-------|------------|------------|--------------|----------------|
| `Cohere embed-v4` | 65.2 | 1024 | 100+ languages | $0.10 |
| `text-embedding-3-large` | 64.6 | 3072 | Strong | $0.13 |
| `voyage-3-large` | 63.8 | 1536 | Good | $0.12 |
| `nomic-embed-text-v2` | 59.4 | 768 | 100 languages | Open-source |
| `BGE-M3` | 63.0 | 1024 | 100+ languages | Open-source |

**For SevanBot (Turkish Content):**

```python
# Option 1: OpenAI (good Turkish support, easy integration)
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-large",  # or "text-embedding-3-small" for cost
    input=text,
    dimensions=1024,  # Can reduce dimensions to save storage
)

# Option 2: Cohere (excellent multilingual)
import cohere

co = cohere.Client()
response = co.embed(
    texts=[text],
    model="embed-v4.0",
    input_type="search_document",
)

# Option 3: Open-source (self-hosted, free)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(texts)
```

---

### 3.5 Hybrid Search

**Course Approach (06_hybrid_search.ipynb):**
- BM25 sparse index (`bm25s`)
- Dense vector index (Qdrant)
- Weighted average fusion (0.8 dense + 0.2 sparse)

**Modern Improvements:**

**1. Use Reciprocal Rank Fusion (RRF) instead of weighted scoring:**

```python
def reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """
    Combine multiple ranked lists using RRF.

    Args:
        results_lists: List of ranked result lists (each item has 'id')
        k: Smoothing constant (default 60, from research)

    Returns:
        Fused ranked list with RRF scores
    """
    fused_scores = {}

    for results in results_lists:
        for rank, result in enumerate(results, 1):
            doc_id = result['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'doc': result, 'score': 0}
            fused_scores[doc_id]['score'] += 1 / (rank + k)

    # Sort by fused score
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    return sorted_results
```

**Why RRF over Weighted Scoring:**
- No normalization needed (BM25 and cosine scores have different scales)
- More robust across different data distributions
- Simpler to tune (only one parameter: k)

**2. Use a vector database with native hybrid support:**

```python
# Qdrant with sparse vectors (native hybrid)
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, PointStruct

# Qdrant natively supports hybrid search with RRF
client.search(
    collection_name="documents",
    query_vector=dense_vector,
    sparse_vector=sparse_vector,
    fusion=Fusion.RRF,  # Built-in RRF support
)
```

---

### 3.6 Reranking

**Course Approach (07_reranking.ipynb):**
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores all retrieved documents against query

**Modern Options:**

| Reranker | Type | Latency | Quality |
|----------|------|---------|---------|
| `ms-marco-MiniLM` (course) | Cross-encoder | ~30ms | Good |
| `bge-reranker-v2-m3` | Cross-encoder | ~50ms | Better |
| `Cohere Rerank 3.5` | API | ~100ms | Best |
| `ColBERT v2` | Late interaction | ~25ms | Best (with precomputation) |

**Recommended Implementation:**

```python
# Option 1: Modern cross-encoder (open-source)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

pairs = [[query, doc['text']] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by scores
reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)

# Option 2: Cohere Rerank API (best quality)
import cohere

co = cohere.Client()
results = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=[doc['text'] for doc in retrieved_docs],
    top_n=5,
)

# Option 3: ColBERT via RAGatouille (late interaction)
from ragatouille import RAGPretrainedModel

rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
reranked = rag.rerank(query=query, documents=docs, k=5)
```

---

### 3.7 Query Understanding (NEW - Not in Course)

**Modern addition: Transform queries before retrieval**

```python
from openai import OpenAI

client = OpenAI()

def expand_query(query: str, n_variations: int = 3) -> list:
    """Generate query variations to improve recall."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate {n_variations} different ways to search for the same information as this query. Return only the queries, one per line.

Query: {query}"""
        }],
        temperature=0.7,
    )

    variations = response.choices[0].message.content.strip().split('\n')
    return [query] + variations  # Include original

def decompose_query(query: str) -> list:
    """Break complex query into sub-queries."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""If this query requires multiple pieces of information, break it into simpler sub-queries. If it's already simple, return it as-is. Return one query per line.

Query: {query}"""
        }],
        temperature=0.3,
    )

    sub_queries = response.choices[0].message.content.strip().split('\n')
    return sub_queries
```

---

## 4. Complete Pipeline Implementation

```python
"""
Modern RAG Retrieval Pipeline
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RetrievedDocument:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class ModernRAGRetriever:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        use_query_expansion: bool = True,
    ):
        self.embedding_model = embedding_model
        self.reranker = CrossEncoder(reranker_model)
        self.use_query_expansion = use_query_expansion

        # Initialize vector store and BM25 index
        self.vector_store = None  # Qdrant, Pinecone, etc.
        self.bm25_index = None

    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with semantic chunking and context enrichment."""

        for doc in documents:
            # 1. Semantic chunking
            chunks = self.semantic_chunk(doc['content'])

            # 2. Add context to each chunk
            enriched_chunks = [
                self.add_context(chunk, doc['metadata'])
                for chunk in chunks
            ]

            # 3. Create embeddings
            embeddings = self.embed(enriched_chunks)

            # 4. Index in vector store (dense)
            self.vector_store.upsert(embeddings, enriched_chunks, doc['metadata'])

            # 5. Index in BM25 (sparse)
            self.bm25_index.add(enriched_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ) -> List[RetrievedDocument]:
        """Execute full retrieval pipeline."""

        # 1. Query expansion (optional)
        if self.use_query_expansion:
            queries = self.expand_query(query)
        else:
            queries = [query]

        # 2. Hybrid search for each query
        all_dense_results = []
        all_sparse_results = []

        for q in queries:
            dense = self.vector_store.search(self.embed([q])[0], top_k=top_k)
            sparse = self.bm25_index.search(q, top_k=top_k)
            all_dense_results.extend(dense)
            all_sparse_results.extend(sparse)

        # 3. Reciprocal Rank Fusion
        fused_results = self.rrf_fusion([all_dense_results, all_sparse_results])

        # 4. Rerank top candidates
        candidates = fused_results[:top_k * 2]  # Rerank more than needed
        reranked = self.rerank(query, candidates)

        return reranked[:rerank_top_n]

    def rrf_fusion(self, results_lists: List[List], k: int = 60) -> List:
        """Reciprocal Rank Fusion."""
        scores = {}
        for results in results_lists:
            for rank, doc in enumerate(results, 1):
                doc_id = doc['id']
                if doc_id not in scores:
                    scores[doc_id] = {'doc': doc, 'score': 0}
                scores[doc_id]['score'] += 1 / (rank + k)

        return sorted(scores.values(), key=lambda x: x['score'], reverse=True)

    def rerank(self, query: str, docs: List) -> List:
        """Rerank using cross-encoder."""
        pairs = [[query, d['doc']['text']] for d in docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(docs, scores):
            doc['rerank_score'] = score

        return sorted(docs, key=lambda x: x['rerank_score'], reverse=True)
```

---

## 5. Recommended Implementation Order

### Phase 1: Foundation (Start Here)
1. Set up semantic chunking with your existing documents
2. Choose and integrate embedding model (`text-embedding-3-large` recommended)
3. Set up vector database (Qdrant or similar)
4. Implement basic dense retrieval

### Phase 2: Hybrid Search
5. Add BM25 sparse indexing
6. Implement RRF fusion for hybrid results
7. Test and compare against dense-only

### Phase 3: Reranking
8. Add cross-encoder reranking
9. Tune retrieval count (retrieve more, rerank to fewer)

### Phase 4: Advanced (Optional)
10. Add query expansion/decomposition
11. Experiment with contextual embeddings
12. Consider ColBERT for high-volume scenarios

---

## 6. Key Metrics to Track

| Metric | What it Measures | Target |
|--------|------------------|--------|
| **Recall@K** | % of relevant docs in top K | >90% at K=10 |
| **MRR** | Mean Reciprocal Rank | >0.7 |
| **Latency** | End-to-end retrieval time | <500ms |
| **Index Size** | Storage per document | Monitor growth |

---

## 7. Sources

### Semantic Chunking
- [Best Chunking Strategies for RAG in 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Smarter RAG with Semantic Chunking](https://milvus.io/blog/embedding-first-chunking-second-smarter-rag-retrieval-with-max-min-semantic-chunking.md)

### Contextual Retrieval & Late Chunking
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Late Chunking Paper (arXiv)](https://arxiv.org/abs/2409.04701)
- [Voyage Context-3 Announcement](https://blog.voyageai.com/2025/07/23/voyage-context-3/)

### Hybrid Search & RRF
- [OpenSearch: Introducing RRF](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Elastic: Weighted RRF](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf)
- [MongoDB: RRF vs RSF](https://medium.com/mongodb/reciprocal-rank-fusion-and-relative-score-fusion-classic-hybrid-search-techniques-3bf91008b81d)

### Reranking & ColBERT
- [Weaviate: Late Interaction Models Overview](https://weaviate.io/blog/late-interaction-overview)
- [ColPali for RAG (2025)](https://medium.com/@intuitivedl/rag-with-colpali-everything-you-need-to-know-46b7bd50901b)
- [ModernBERT + ColBERT (arXiv)](https://arxiv.org/abs/2510.04757)

### Embedding Models
- [Top Embedding Models 2025](https://artsmart.ai/blog/top-embedding-models-in-2025/)
- [MTEB Leaderboard Guide](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)

### Query Understanding
- [Haystack: Query Expansion](https://haystack.deepset.ai/blog/query-expansion)
- [Haystack: Query Decomposition](https://haystack.deepset.ai/cookbook/query_decomposition)
