# Migration Guide: Old RAG → Hybrid Retrieval System

## Prerequisites

Before starting the migration, ensure:

1. ✅ Docker containers are running (ChromaDB on port 8000)
2. ✅ B2 bucket credentials are configured
3. ✅ At least 10GB free disk space for models and indices
4. ✅ Python 3.11+ and Poetry installed

## Step-by-Step Migration

### Step 1: Install New Dependencies

```bash
cd /home/adi/Work/ai.dok/llm_interaction_service

# Update dependencies using Poetry
poetry install

# This will install:
# - rank-bm25==0.2.2
# - sentence-transformers==2.3.0 (updated)
# - transformers==4.36.0 (updated)
# - torch==2.1.0 (if not already installed)
```

**Expected time**: 5-10 minutes (depends on internet speed)

### Step 2: Download Embedding Models (One-Time Setup)

Models will be downloaded automatically on first use, but you can pre-download them:

```bash
cd /home/adi/Work/ai.dok/llm_interaction_service

# Pre-download models (optional but recommended)
poetry run python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Downloading BGE-Large embeddings...')
SentenceTransformer('BAAI/bge-large-en-v1.5')
print('Downloading BGE-Reranker...')
CrossEncoder('BAAI/bge-reranker-large')
print('Models downloaded successfully!')
"
```

**Model sizes**:
- `BAAI/bge-large-en-v1.5`: ~1.3 GB
- `BAAI/bge-reranker-large`: ~1.1 GB

**Expected time**: 10-15 minutes

### Step 3: Create Data Directory for BM25 Index

```bash
cd /home/adi/Work/ai.dok/llm_interaction_service
mkdir -p data
```

### Step 4: Update Environment Variables (Optional)

Add to `.env` file (all have defaults, so this is optional):

```bash
# Embedding and Reranking Models
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large

# Retrieval Strategy (default: hybrid)
RETRIEVAL_STRATEGY=hybrid

# BM25 Parameters
BM25_K1=1.5
BM25_B=0.75

# Hybrid Retrieval Parameters
HYBRID_ALPHA=0.5        # 0=BM25 only, 1=vector only
RETRIEVAL_TOP_K=20      # Initial candidates before reranking
RERANKER_TOP_K=5        # Final results after reranking

# RCP Section Chunking
CHUNK_BY_SECTION=true
CHUNK_SIZE=512
CHUNK_OVERLAP=100
```

### Step 5: Restart Services

```bash
cd /home/adi/Work/ai.dok

# Restart the LLM interaction service
docker-compose restart llm_interaction_service

# Or rebuild if needed
docker-compose up -d --build llm_interaction_service
```

### Step 6: Index PDFs to New Collection

**Option A: Via Python Script**

```python
# create_index.py
import asyncio
from app.services.indexing_service import IndexingService

async def main():
    print("Starting indexing with hybrid retrieval system...")
    
    # Create indexing service for new collection
    indexing_service = IndexingService(
        collection_name="rcp_documents_v2",
        use_section_chunking=True
    )
    
    # Process all PDFs from B2 bucket
    result = await indexing_service.process_bucket()
    
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Collection: {result['collection_name']}")
    print(f"Processed PDFs: {result['processed_pdf_files']}/{result['total_pdf_files_in_bucket']}")
    print(f"Total Chunks: {result['total_chunks']}")
    print(f"BM25 Corpus Size: {result['bm25_corpus_size']}")
    print(f"Failed Files: {len(result['failed_files'])}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
cd /home/adi/Work/ai.dok/llm_interaction_service
poetry run python create_index.py
```

**Option B: Via API Endpoint**

```bash
curl -X POST "http://localhost:9322/indexing/process-bucket" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected time**: 12-24 hours for 6000 PDFs (depends on OCR needs)

**Progress monitoring**: Check logs for progress
```bash
docker logs -f llm_interaction_service
```

### Step 7: Verify Indexing

Check that both collections exist:

```bash
# Via Python
poetry run python -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
collections = client.list_collections()
print('Available collections:')
for col in collections:
    print(f'  - {col.name}')
"
```

Expected output:
```
Available collections:
  - rcp_documents (old - MiniLM)
  - rcp_documents_v2 (new - BGE)
```

Verify BM25 index:
```bash
ls -lh /home/adi/Work/ai.dok/llm_interaction_service/data/
# Should show: bm25_index_rcp_documents_v2.pkl (~180 MB)
```

### Step 8: Test Query with New System

```bash
curl -X POST "http://localhost:9322/llm-interaction-api/v1/rag-pipeline" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=Care sunt contraindicațiile pentru 5-Fluorouracil?" \
  -F "model=llama-3.3-70b-versatile" \
  -F "ai_service=groq_cloud" \
  -F "collection_name=rcp_documents_v2" \
  -F "retrieval_strategy=hybrid" \
  -F "top_k=5"
```

**Expected response structure**:
```json
{
  "response": "<h3>Contraindicații</h3><p>Conform secțiunii 4.3...</p>",
  "retrieved_documents": [
    {
      "page_content": "...",
      "metadata": {
        "source": "5-Fluorouracil-Ebewe.pdf",
        "section_number": "4.3",
        "section_title": "CONTRAINDICAȚII"
      },
      "relevance_score": 0.89
    }
  ],
  "retrieval_strategy": "hybrid",
  "num_documents_retrieved": 5,
  "low_confidence": false
}
```

### Step 9: Run Evaluation (Optional but Recommended)

```bash
cd /home/adi/Work/ai.dok/llm_interaction_service

# Run evaluation on test set
poetry run python tests/rag_evaluator.py

# Expected output:
# ============================================================
# Results for HYBRID
# ============================================================
# Total Queries:        50
# Precision@5:          0.847
# Recall@5:             0.920
# MRR:                  0.883
# Hallucination Rate:   0.12
# Section Accuracy:     0.91
# Avg Response Time:    2.3s
# ============================================================
```

### Step 10: Benchmark Strategies (Compare Old vs New)

```bash
curl -X POST "http://localhost:9322/llm-interaction-api/v1/rag-pipeline/benchmark" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "prompt=Ce reacții adverse poate cauza 5-Fluorouracil?" \
  -F "model=llama-3.3-70b-versatile"
```

This will compare:
- `vector_only` (old approach)
- `bm25_only` (keyword only)
- `hybrid` (new approach)

### Step 11: Switch to Production

Once satisfied with testing:

**Update default in code** (already done in implementation):
- Default collection: `rcp_documents_v2`
- Default strategy: `hybrid`

**Or override via environment**:
```bash
# In .env
RETRIEVAL_STRATEGY=hybrid
```

### Step 12: Monitor Production

```bash
# Watch logs
docker logs -f llm_interaction_service | grep "RAG query"

# Check metrics
# - Response times
# - Low confidence flags
# - User feedback
```

### Step 13: Cleanup (After 1 Month Validation)

Once new system is validated:

```python
# Delete old collection (optional - keep as backup)
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
client.delete_collection("rcp_documents")
```

---

## Rollback Procedure

If issues arise with the new system:

### Quick Rollback (Endpoint Level)

```bash
# Override in API calls
collection_name=rcp_documents  # Use old collection
retrieval_strategy=vector_only  # Use old strategy
```

### Full Rollback (Code Level)

1. Revert changes in `app/core/constants.py`:
```python
class ChromaCollection(str, Enum):
    RCP_DOCUMENTS = "rcp_documents"  # Change default back
```

2. Revert default strategy in `app/core/config.py`:
```python
retrieval_strategy: str = Field(default='vector_only', ...)
```

3. Restart service:
```bash
docker-compose restart llm_interaction_service
```

---

## Troubleshooting

### Issue: "Collection rcp_documents_v2 not found"

**Cause**: Indexing not completed

**Solution**:
```bash
# Check collection exists
poetry run python -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
print([c.name for c in client.list_collections()])
"

# If missing, run indexing (Step 6)
```

### Issue: "BM25 index not found"

**Cause**: BM25 index file missing or corrupted

**Solution**:
```bash
# Check file exists
ls -lh data/bm25_index_rcp_documents_v2.pkl

# If missing, rebuild by re-running indexing
# The BM25 index is built automatically during indexing
```

### Issue: Model download fails

**Cause**: No internet connection or HuggingFace Hub down

**Solution**:
```bash
# Pre-download models manually
wget https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/pytorch_model.bin
wget https://huggingface.co/BAAI/bge-reranker-large/resolve/main/pytorch_model.bin

# Or use offline mode if models are cached
export TRANSFORMERS_OFFLINE=1
```

### Issue: Slow queries (>10 seconds)

**Symptoms**: Hybrid retrieval takes too long

**Solution**:
```bash
# Reduce retrieval_top_k in .env
RETRIEVAL_TOP_K=10  # Instead of 20

# Or use vector_only for faster queries
retrieval_strategy=vector_only
```

### Issue: Section parsing failures

**Symptoms**: Many documents show "No sections found, using fallback"

**Check**:
```python
from app.services.rcp_section_parser_service import RCPSectionParserService
parser = RCPSectionParserService()

# Test on sample RCP
with open('sample_rcp.txt') as f:
    text = f.read()
    sections = parser.parse_sections(text)
    print(f"Found {len(sections)} sections")
    for s in sections:
        print(f"  {s.number}: {s.title}")
```

---

## Validation Checklist

Before considering migration complete:

- [ ] All 6000 PDFs indexed to `rcp_documents_v2`
- [ ] BM25 index built successfully
- [ ] Test queries return relevant results
- [ ] Section citations appear in responses
- [ ] Hallucination guardrails trigger when appropriate
- [ ] Response times acceptable (<5s including LLM)
- [ ] Evaluation metrics meet targets:
  - [ ] Precision@5 > 0.80
  - [ ] Recall@5 > 0.85
  - [ ] Hallucination rate < 0.15
  - [ ] Section accuracy > 0.85
- [ ] Benchmark shows improvement over old system
- [ ] No errors in production logs for 1 week
- [ ] Positive feedback from test users

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| **Install dependencies** | 15 min | ✅ Ready |
| **Download models** | 15 min | ⏳ Pending |
| **Index 6000 PDFs** | 12-24 hours | ⏳ Pending |
| **Testing & validation** | 3-5 days | ⏳ Pending |
| **Production rollout** | 1 day | ⏳ Pending |
| **Monitoring period** | 1 month | ⏳ Pending |
| **Total** | ~1 month | In Progress |

---

## Support

For issues during migration:
- Check logs: `docker logs -f llm_interaction_service`
- Review documentation: `HYBRID_RETRIEVAL_README.md`
- Test individual components with Python scripts
- GitHub Issues: [alexban14/ai.dok](https://github.com/alexban14/ai.dok)

**Last Updated**: November 18, 2025
