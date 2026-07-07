# Project Report – PDF Question Answering RAG Chatbot

## Summary

The PDF Question Answering RAG Chatbot is a portfolio project that demonstrates how to build a document‑grounded question answering system without external APIs.  Users upload PDF documents and ask questions about their content.  The application extracts and indexes the text, retrieves relevant passages and produces an answer that cites the sources.  The system is implemented in Python with a modular architecture and includes both a Streamlit web interface and a command‑line interface.

## Architecture

The project follows a layered architecture aligned with the RAG paradigm:

1. **Data ingestion** – PDFs are loaded and parsed with `pypdf` in `src/pdf_loader.py`.  Text is extracted on a per‑page basis.
2. **Pre‑processing** – Text is split into overlapping chunks by `src/text_splitter.py`.  Each chunk retains metadata (document name, page number, chunk id).
3. **Embedding** – `src/embeddings.py` encodes each chunk into a numeric vector.  The default backend uses a semantic SentenceTransformer model ("all‑MiniLM‑L6‑v2") to capture the meaning of text.  If the model cannot be loaded (for example, due to missing dependencies or offline environments), the system falls back to a TF‑IDF vectorizer.
4. **Storage** – Embeddings and metadata are stored in an in‑memory vector store implemented in `src/vector_store.py`.  This module also exposes a cosine similarity search function.
5. **Retrieval** – `src/retriever.py` wraps the vector store and returns the top‑k most similar chunks to a query.
6. **Answer generation** – `src/answer_generator.py` synthesises a short answer from the retrieved chunks and returns the supporting passages.
7. **Orchestration** – The entire workflow is coordinated by `src/rag_pipeline.py`, which exposes a simple API for loading documents, asking questions and receiving answers with citations.

## Pipeline

The end‑to‑end pipeline proceeds as follows:

1. The user uploads one or more PDF files via the Streamlit app or passes a path via the CLI.
2. The pipeline loads each PDF using the `PDFLoader` class, skipping files that cannot be read or do not contain extractable text.
3. The `TextSplitter` class divides each page’s text into overlapping chunks.  Overlap ensures that information near the boundaries is not lost.
4. The embedding component is initialised via a factory.  It attempts to load the semantic model and fit on the corpus of chunk texts.  If this fails, it falls back to fitting a TF‑IDF vectorizer.  Both chunks and queries are encoded into the same vector space.
5. The `VectorStore` stores all chunk vectors and supports cosine similarity search.  Each stored item includes the chunk text and its metadata.
6. When a question is asked, its embedding is computed and the `Retriever` returns the top‑k chunks with the highest similarity scores.
7. The `AnswerGenerator` analyses the retrieved chunks, extracts relevant sentences and constructs an answer.  If the retrieved context does not cover the question, a fallback message is returned.

## Main Modules

* **`pdf_loader.py`** – Provides a `PDFLoader` class that wraps `pypdf.PdfReader`.  It iterates over pages and yields dictionaries containing the text and metadata.  Handles missing files, empty documents, encrypted PDFs and scanned PDFs (with no extractable text) gracefully.
* **`text_splitter.py`** – Implements the `TextSplitter` class.  It checks the validity of `chunk_size` and `chunk_overlap` parameters and splits text using a sliding window strategy.  The output is a list of chunk dictionaries.
* **`embeddings.py`** – Provides two concrete classes, `SentenceTransformerEmbeddings` and `TfidfEmbeddings`, and a factory function to select between them.  The wrapper class `EmbeddingModel` retains backwards compatibility.  By default a semantic model is used; TF‑IDF serves as a fallback.
* **`vector_store.py`** – Contains `VectorStore`, which stores vectors in a NumPy array and associated metadata in a list.  It offers a `query` method that computes cosine similarities and returns the top‑k results.
* **`retriever.py`** – A thin wrapper around the vector store that exposes a `retrieve` method with a configurable `top_k`.
* **`answer_generator.py`** – Implements a simple extractive answer generator.  It searches within retrieved chunks for sentences that mention keywords from the question and concatenates them into a coherent response.  If no such sentences exist, it triggers the fallback message.
* **`rag_pipeline.py`** – Coordinates the above components.  It exposes `load_documents` and `ask` methods and maintains the vector store and embedding model between calls.

## Testing Strategy

The project includes a suite of pytest tests located in the `tests/` directory:

* **Text splitter tests** ensure that the splitter respects the configured sizes, rejects invalid parameters and preserves metadata.
* **PDF loader tests** create small PDFs on the fly using `reportlab` and verify that text extraction returns the correct number of pages and metadata.
* **Retriever tests** use fake embeddings to verify that the retrieval logic returns the expected chunks in order of decreasing similarity.
* **Pipeline tests** validate that the pipeline handles empty documents gracefully and returns the fallback message when context is insufficient.

These tests do not require internet access or API keys and run quickly.

## Limitations

* The system does not perform OCR and thus cannot handle scanned documents.
* The system falls back to TF‑IDF embeddings when the semantic model cannot be loaded.  TF‑IDF provides only lexical similarity and may miss paraphrased information.
* Answers are assembled by extracting sentences from retrieved chunks; there is no generative language model.  The phrasing may therefore be rigid.
* The vector store is held in memory and not persisted to disk.

## Future Work

Potential improvements include adding OCR support, persisting the vector store to disk, incorporating a lightweight generative model for answer synthesis and extending the user interface with additional features such as multi‑question chat history.