# Plan: AI Model Development

This document outlines the plan for developing an AI model to assist doctors by analyzing patient prescriptions against a knowledge base of ~31,000 RCP documents. The focus is on providing the **simplest, most effective, and reliable solution**, which is **Retrieval-Augmented Generation (RAG)**, not fine-tuning.

### Why RAG over Fine-Tuning?

*   **Simplicity & Cost:** Fine-tuning a large language model is computationally expensive and complex. A RAG system is significantly simpler to build and maintain.
*   **Accuracy & Reduced Hallucination:** Fine-tuning bakes knowledge into the model's weights, which can lead to "hallucination" (making things up). RAG is grounded in the actual text from the RCPs provided at query time, making it far more factual and trustworthy.
*   **Updatability:** When a new drug or updated RCP is released, you simply add its processed text to the knowledge base. With fine-tuning, you would need to re-run the entire training process.
*   **Explainability:** A RAG system can cite its sources, showing the doctor the exact text snippets from the RCPs that were used to generate the summary. This is critical for medical applications.

## The RAG Architecture

The system will have two main stages: an offline **Indexing Pipeline** and an online **Query Pipeline**.

### 1. Indexing Pipeline (Offline Process)

**Objective:** To convert the structured JSON data from the RCPs into a searchable vector knowledge base.

**Technology Stack:**
*   **Vector Embeddings Model:** A sentence-transformer model like `all-MiniLM-L6-v2` (fast and effective) or a more powerful model from the Hugging Face Hub.
*   **Vector Database:**
    *   `ChromaDB` or `FAISS`: Excellent open-source options that can be run locally. They are perfect for a PhD project.
    *   `Pinecone` or `Weaviate`: Cloud-based solutions if scalability becomes a major factor.
*   **Programming Language:** Python

**Steps:**

1.  **Data Loading:** Load the structured JSON files created by the data extraction bot.
2.  **Text Chunking:** The text from each section (e.g., "adverse_reactions") needs to be split into smaller, semantically meaningful chunks (e.g., paragraphs or sentences). This is crucial for the retrieval to be precise. A chunk should not be too large.
3.  **Embedding Generation:** For each text chunk, use the sentence-transformer model to generate a vector embedding (a numerical representation of the text's meaning).
4.  **Indexing:** Store each text chunk and its corresponding vector embedding in the chosen vector database. The chunk should also have metadata linking it back to the original drug name and RCP section (e.g., `{"drug": "ExampleDrug", "section": "interactions"}`).

### 2. Query Pipeline (Online Process)

**Objective:** To take a doctor's query, retrieve relevant information from the knowledge base, and generate a concise, accurate summary.

**Technology Stack:**
*   **Large Language Model (LLM):**
    *   **Open Source:** `Llama 3`, `Mixtral`, or other powerful instruction-following models. These can be run locally on appropriate hardware.
    *   **API-based:** OpenAI's GPT-4, Google's Gemini, or Anthropic's Claude 3. These are very powerful and easy to use via API calls.
*   **Framework:** `LangChain` or `LlamaIndex` to orchestrate the RAG pipeline.

**Steps:**

1.  **Input Processing:** The system takes the patient's information as input. This includes:
    *   List of prescribed drugs (e.g., ["Drug A", "Drug B", "Drug C"]).
    *   Patient conditions (e.g., "is pregnant").

2.  **Multi-Query Generation:** The input is broken down into a series of specific questions that need to be answered by the system:
    *   "What are the interactions between Drug A and Drug B?"
    *   "What are the interactions between Drug A and Drug C?"
    *   "What are the common adverse reactions of Drug A?"
    *   "What are the common adverse reactions of Drug B?"
    *   "Is Drug A contraindicated for pregnant women?"
    *   ...and so on.

3.  **Retrieval:** For each sub-question, the system:
    *   Creates a vector embedding of the question.
    *   Queries the vector database to find the most similar text chunks from the RCPs. For an interaction query, it would retrieve chunks from the "interactions" sections of the relevant drugs.

4.  **Augmentation & Generation:**
    *   The retrieved text chunks (the "context") are combined with the original sub-question into a detailed prompt for the LLM.
    *   **Prompt Engineering is key here.** The prompt will instruct the LLM to act as a medical expert and answer the question *based only on the provided context*.
    *   Example Prompt:
        ```
        [CONTEXT]
        From Drug A RCP, section 4.5: "Drug A should not be used with..."
        From Drug B RCP, section 4.5: "Caution is advised when co-administering with..."
        ---
        [QUESTION]
        Based only on the context provided, what are the interactions between Drug A and Drug B?
        ```
    *   The LLM generates an answer for each sub-question.

5.  **Final Summary:**
    *   The answers to all the sub-questions are collected.
    *   A final prompt is sent to the LLM, asking it to synthesize all the individual answers into a single, clear, and concise summary for the doctor, highlighting the most critical information (e.g., severe interactions, overlapping adverse reactions).

### 3. User Interface

A simple web interface can be built using **Streamlit** or **FastAPI + a simple frontend** to allow the doctor to input the patient data and view the generated summary.
