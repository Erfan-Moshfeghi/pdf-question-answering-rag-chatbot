# Interview Notes: Understanding the PDF RAG Chatbot

This document provides conceptual explanations of the key ideas implemented in the PDF Question Answering RAG Chatbot.  If you are presenting this project in an interview, these notes can help articulate the technical rationale and design choices.

## What is Retrieval Augmented Generation (RAG)?

*RAG* is a hybrid approach that combines information retrieval with generative models.  Rather than generating answers from a static model alone, a RAG system first **retrieves** relevant documents or fragments from an external knowledge base, then uses those retrieved snippets to guide or condition the **generation** of a response.  This approach reduces hallucination and allows the system to answer questions about content that was not seen during model training.

In the context of this project, the knowledge base consists of the PDF files uploaded by the user.  We build an index of the documents and retrieve the most relevant chunks for a given question.

## Why do we split text into chunks?

Large documents cannot be indexed effectively at the document level because relevant information might be buried in a long section.  Splitting text into smaller, overlapping chunks improves granularity.  Overlapping windows ensure that context near the boundaries of chunks is preserved.  For example, with a chunk size of 800 characters and an overlap of 150 characters, each new chunk starts 650 characters after the previous one.  This overlap means that sentences spanning two chunks are still represented in at least one chunk.

## What are embeddings?

An **embedding** converts a piece of text into a numeric vector so that we can measure similarity between texts using mathematical operations.  There are two broad approaches implemented in this project:

* **Semantic embeddings** – A pre‑trained SentenceTransformer model ("all‑MiniLM‑L6‑v2") produces dense vectors that capture the meaning of words and phrases.  These embeddings understand context and synonyms, making them ideal for RAG systems.  They are now the **default** backend in this project.
* **Keyword embeddings (TF–IDF)** – The TF–IDF (term frequency–inverse document frequency) vectorizer from scikit‑learn represents documents based on the frequency of each term scaled by how rare the term is across all documents.  It is lexical and does not capture semantics.  This backend serves as a lightweight fallback for environments where semantic models cannot be loaded (e.g. offline or constrained resources).

The embedding API hides the details of which backend is in use, but understanding the distinction helps explain differences in retrieval behaviour.

## How does vector search work?

Once we have embeddings for all chunks and the user’s query, we can perform **vector search**.  We compute the **cosine similarity** between the query vector and each chunk vector, which measures the angle between the vectors.  A value closer to 1 indicates that the texts are more similar.  We then select the top‑k most similar chunks as the relevant context for answering the question.

## Source‑grounded answers and hallucination reduction

In typical chatbots, the model might attempt to answer questions even if it has no knowledge of the subject, leading to hallucinations.  A RAG chatbot mitigates this by strictly conditioning its answer on the retrieved context.  If the system cannot find relevant chunks, it returns a message indicating that the answer is not present in the documents.  This transparency is crucial for trustworthiness.

## Project limitations

* **No OCR**: The PDF loader only extracts embedded text.  Scanned documents or images are ignored because optical character recognition is not implemented.  You could integrate libraries such as Tesseract to support scanned PDFs.
* **Embedding model availability**: Although semantic embeddings are the default, they depend on the `sentence-transformers` library and a pre‑trained model.  In offline environments the system falls back to TF–IDF, which lacks semantic understanding and may limit performance on complex queries.
* **Extractive answering**: The current answer generator combines sentences from the retrieved chunks without using a language model.  While this avoids hallucination, the phrasing can be less natural than LLM‑generated answers.
* **Local state**: The vector store and session data are stored in memory.  In a production system you would persist the index to disk or a database for reuse across sessions.

## Conclusion

This project showcases how to build a document question answering system from scratch using only open‑source Python libraries.  It emphasises modularity, testability and explainability, making it an excellent portfolio piece for demonstrating understanding of RAG, embeddings and information retrieval.