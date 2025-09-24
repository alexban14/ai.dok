# Concept: Text vs. Vector Embeddings for RAG

This document explains the core concepts behind using vector embeddings for Retrieval-Augmented Generation (RAG) and why it is the ideal approach for the AI-Dok project.

## Core Concepts

### 1. Raw Text
*   **What it is:** A sequence of characters (letters, numbers, punctuation) that humans can read.
*   **How a computer sees it:** As data to be stored and retrieved. It has no inherent understanding of the *meaning* behind the words "annual report" vs. "yearly summary." They are just different strings.
*   **Searching:** You can only perform literal, keyword-based searches (like `Ctrl+F`). You have to match the exact characters.

### 2. Vector Embeddings
*   **What it is:** A list of numbers (a vector) that represents the *semantic meaning* of a piece of text.
*   **How it's made:** A specialized AI model (in our case, `all-MiniLM-L6-v2`) reads a piece of text and converts its conceptual meaning into a dense vector of numbers.
*   **The "Magic":** The embedding model is trained to place text with similar meanings close to each other in a high-dimensional "meaning space."
    *   **Analogy:** Imagine a giant library where books are organized not by title, but by their core concepts. A book about "corporate finance" would be on the same shelf as a book about "business accounting," even if they don't share the exact same words. A vector embedding is like the book's precise coordinate on a shelf in that library.

---

## Comparing Approaches for Document Analysis

The goal is to answer a user's question based on the content of an RCP document.

### Approach 1: Search over Raw Text (The Inefficient, Brittle Way)

1.  You would store all the extracted text chunks in a regular database.
2.  A user asks: "What was the company's revenue last year?"
3.  Your system would have to perform a keyword search for "revenue" and "last year" across all text chunks.
4.  **Problem:** What if the document says "annual earnings" or "total income for the previous fiscal period"? A keyword search would completely miss this. It has no conceptual understanding.

### Approach 2: Search over Vector Embeddings (The Better, Semantic Way)

This is the standard approach for modern RAG pipelines.

#### Part A: Indexing (Offline Process)
1.  **Extract & Chunk:** Take the text from a PDF and split it into manageable chunks.
2.  **Vectorize:** For each chunk of text, use the embedding model to create a vector embedding.
3.  **Store:** Store this vector in ChromaDB. **Crucially, you also store the original text chunk as metadata alongside its vector.** ChromaDB is designed for exactly this: linking a vector to its source data.

#### Part B: Querying (Online Process)
1.  **Vectorize the Question:** A user submits a prompt, e.g., "What was the company's revenue last year?". Your application takes this question and uses the *same embedding model* to turn it into a "question vector."
2.  **Similarity Search:** You then query ChromaDB with this question vector. ChromaDB doesn't look for matching words. Instead, it calculates the "distance" between your question vector and all the vectors stored in the database. It finds the text chunks whose vectors are *semantically closest* to the user's question.
3.  **Retrieve Context:** ChromaDB returns the top N most similar chunks of original text (the metadata). These chunks will likely contain phrases like "annual earnings" or "total income," because their *meaning* is close to "last year's revenue."
4.  **Generate Answer:** You then feed these retrieved text chunks to your generative LLM (Groq/Ollama) with a final prompt like:
    > "Based on the following context from the document, please answer the user's question.
    >
    > **Context:** [Insert the retrieved text chunks here]
    >
    > **User's Question:** What was the company's revenue last year?"

### Conclusion: The Better Approach

The **vector embedding approach is vastly superior** and is the correct one for your use case.

*   **It finds answers based on meaning, not just keywords.** This handles synonyms, paraphrasing, and related concepts, making your search incredibly robust.
*   **It's efficient.** Vector similarity search is extremely fast, even with millions of documents.
*   **It yields better results.** By providing the LLM with highly relevant, factual context pulled directly from the source document, you dramatically improve the accuracy of the final answer and prevent the model from making things up (hallucinating).
