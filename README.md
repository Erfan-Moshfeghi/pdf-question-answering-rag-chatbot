# PDF Question Answering RAG Chatbot

Welcome to the **PDF Question Answering RAG Chatbot** project.  This repository contains a complete, portfolio‑ready Retrieval Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their contents.  The chatbot extracts text from your PDFs, indexes the information into a vector store and retrieves the most relevant chunks to generate a source‑grounded answer.  It is designed as an educational project suitable for internship applications and technical interviews.

## Overview

This project demonstrates a clean and modular RAG pipeline implemented in pure Python.  Users can interact with the system either through a **Streamlit** web interface or via a simple command‑line interface.  The pipeline performs the following high‑level steps:

1. **PDF Upload** – Upload one or more PDF files via the Streamlit app or specify a PDF path on the command line.
2. **Text Extraction** – Extract text from each page of the PDF using the `pypdf` library while preserving page‑level metadata.
3. **Chunking** – Split the extracted text into overlapping chunks to improve retrieval granularity.  Overlapping windows help the model recall context across chunk boundaries.
4. **Embedding** – Convert each chunk into a high dimensional vector representation.  The default implementation uses TF–IDF vectors via scikit‑learn; you can swap in more advanced models later.
5. **Vector Storage** – Store the embeddings and associated metadata in a simple local vector store.  By default the repository uses an in‑memory implementation to avoid heavy dependencies.
6. **Retrieval** – Given a user question, compute its embedding and perform a similarity search over the stored vectors.  The top‑k most relevant chunks are returned.
7. **Answer Generation** – Generate a concise answer using only the retrieved chunks.  The system returns both the answer and the corresponding source passages with page numbers and document names.  When the answer cannot be found, the user is informed that the information is not in the uploaded documents.

This end‑to‑end flow illustrates how to build an LLM‑style document question answering system without requiring any external APIs.  The code is written to be easy to read, well documented and extensible.

## What problem does it solve?

Large language models are excellent at generating fluent text, but they often **hallucinate** or invent facts.  Retrieval Augmented Generation (RAG) mitigates this by grounding responses in a knowledge base.  In this project the knowledge base is built from user‑uploaded PDFs.  By retrieving relevant passages from the documents and restricting the answer to that context, the chatbot provides answers that are factual and traceable back to the sources.  If the necessary information is not present, the bot will tell you instead of making something up.

## What is RAG?

Retrieval Augmented Generation is a technique that combines information retrieval with language generation.  Rather than relying solely on a pre‑trained model, the system first retrieves relevant documents or document fragments from an external source (in this case, your PDFs).  These retrieved snippets are then passed to a generative model to produce a response.  RAG reduces hallucination and allows the model to answer questions about content it has never seen during training.  For more details on RAG, refer to the accompanying documentation in [`docs/interview_notes.md`](docs/interview_notes.md).

## Key Features

* **Upload multiple PDFs**: Load one or more documents into the system via the UI or CLI.
* **Chunk and index text**: Extract text page by page and split it into overlapping chunks to preserve context.
* **Semantic embeddings**: By default the system uses a pre‑trained SentenceTransformer model ("all‑MiniLM‑L6‑v2") to convert text into dense vectors that capture meaning beyond exact keywords.
* **TF–IDF fallback**: When semantic embeddings cannot be loaded (e.g. in offline environments), the system automatically falls back to a scikit‑learn TF‑IDF vectorizer as a lightweight baseline.
* **Local vector store**: Store vectors and metadata in memory for fast retrieval.  No external database is required.
* **Source‑grounded answers**: Generate answers only from the retrieved passages, with citations including page numbers and document names.
* **Graceful error handling**: Inform the user when PDFs are empty, unreadable or scanned.  If a question cannot be answered from the documents, the system will say so.
* **Streamlit app**: A friendly web interface with adjustable chunk size, overlap and top‑k settings.  Uses `session_state` to persist processed data during the session.
* **Command‑line interface**: A minimal CLI (`main.py`) for running the pipeline outside the browser.
* **Tests**: Pytest tests validate text splitting, metadata preservation, retrieval ranking and pipeline behaviour on empty documents.

## Project Pipeline

```text
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → Answer Generation
```

1. **PDF Loader** (`src/pdf_loader.py`)
   * Uses `pypdf` to read PDF files.
   * Extracts text page by page while recording the document name and page number.
   * Handles missing or unreadable files gracefully.

2. **Text Splitter** (`src/text_splitter.py`)
   * Splits page text into overlapping chunks (default size 800 characters with 150 characters overlap).
   * Preserves document name, page number and chunk id for each piece.
   * Validates parameters to avoid invalid overlap settings.

3. **Embeddings** (`src/embeddings.py`)
   * Provides two implementations: **`SentenceTransformerEmbeddings`** for semantic vectors and **`TfidfEmbeddings`** for lexical vectors.
   * The default configuration uses a SentenceTransformer model ("all‑MiniLM‑L6‑v2").  If the dependency or model is not available, the code gracefully falls back to TF‑IDF.
   * A factory function hides model selection from the rest of the pipeline, allowing easy extension or replacement of the embedding backend.

4. **Vector Store** (`src/vector_store.py`)
   * Stores chunk embeddings and associated metadata in memory.
   * Performs cosine similarity search to retrieve the top‑k most relevant chunks for a query.

5. **Retriever** (`src/retriever.py`)
   * Thin wrapper around the vector store; returns the top‑k results with similarity scores.

6. **Answer Generator** (`src/answer_generator.py`)
   * Constructs a concise answer from retrieved chunks.
   * If no relevant chunks are found, returns a friendly message indicating insufficient context.

7. **RAG Pipeline** (`src/rag_pipeline.py`)
   * Orchestrates the entire workflow.
   * Accepts configuration parameters (chunk size, overlap, top_k, etc.).
   * Returns structured outputs including the answer and the list of source snippets.

## Tech Stack

* **Python 3.10+**
* **Streamlit** for the web interface
* **pypdf** for PDF parsing
* **sentence‑transformers** for semantic embeddings
* **Scikit‑Learn** for TF–IDF fallback embeddings
* **NumPy** for vector operations
* **pytest** for unit testing
* **python‑dotenv** (optional) for future API key management

## Folder Structure

```
pdf-question-answering-rag-chatbot/
├── README.md                # English project overview (this file)
├── README.fa.md             # Persian version of the README
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and directories
├── LICENSE                  # MIT License
├── app.py                   # Streamlit application
├── main.py                  # Command‑line entry point
│
├── src/                     # Core Python package
│   ├── __init__.py          # Makes src a package
│   ├── config.py            # Central configuration values
│   ├── pdf_loader.py        # Load and parse PDFs
│   ├── text_splitter.py     # Split text into chunks
│   ├── embeddings.py        # Compute embeddings
│   ├── vector_store.py      # Simple vector store with cosine similarity
│   ├── retriever.py         # Top‑k retrieval logic
│   ├── answer_generator.py  # Generate answers from retrieved chunks
│   ├── rag_pipeline.py      # Glue all components together
│   └── utils.py             # Miscellaneous helpers
│
├── tests/                   # Unit tests run by pytest
│   ├── conftest.py          # Test fixtures
│   ├── test_text_splitter.py
│   ├── test_pdf_loader.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
│
├── data/                    # Placeholders for data
│   ├── README.md            # Description of the data folder
│   ├── sample_pdfs/         # Example PDFs for development (empty by default)
│   │   └── .gitkeep
│   ├── uploads/             # User uploaded PDFs during app runtime (gitignored)
│   │   └── .gitkeep
│   └── vector_store/        # Vector store files (gitignored)
│       └── .gitkeep
│
├── assets/                  # Optional assets like images or diagrams
│   └── README.md            # Explain the purpose of the assets folder
│
├── docs/                    # Additional documentation
│   └── interview_notes.md   # High level explanations for interviews
│
└── reports/
    └── project_report.md    # Formal project report
```

## Installation (Windows 11)

1. **Clone the repository** (if working from a GitHub clone):

   ```bash
   git clone https://github.com/Erfan-Moshfeghi/pdf-question-answering-rag-chatbot.git
   cd pdf-question-answering-rag-chatbot
   ```

2. **Create a virtual environment** (highly recommended):

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

The default configuration runs entirely locally.  There is no need for API keys or GPU.

> **Note:** On the first run, if semantic embeddings are used and the pre‑trained model is not already cached, the `sentence-transformers` library will attempt to download approximately 90 MB of model weights.  This requires an internet connection.  If the download fails or the dependency is unavailable, the application will automatically fall back to TF‑IDF embeddings.

## Running the Streamlit App

Launch the web application with the following command:

```bash
streamlit run app.py
```

This will open a browser window where you can upload PDFs, set chunk size and overlap, process the documents and ask questions.

## Running the Command‑Line Interface

To process a PDF and ask a single question without the web UI, use the CLI:

```bash
python main.py --pdf path/to/document.pdf --question "What is this document about?"
```

The script will print the answer along with the source snippets.

## Running the Tests

Unit tests are provided to help ensure that the core logic behaves as expected.  To run the tests, first install the requirements and then execute:

```bash
pytest
```

The tests do not require external API calls or large downloads.  They use small dummy inputs and mock embeddings.

## Example Usage

Suppose you upload a PDF manual for a household appliance.  After processing, you can ask:

> **Question:** *How do I reset the device to factory settings?*

The system will search through the indexed chunks, find the section describing the reset procedure and return a concise answer along with the page number and document name.

If the procedure is not described in the manual, the chatbot will reply:

> *I could not find enough information in the uploaded document to answer this question.*

## Limitations

* **Scanned PDFs**: This project does not perform optical character recognition (OCR).  Scanned or image‑only PDFs will not yield extractable text and will therefore be ignored.  Future work could integrate OCR libraries such as Tesseract.
* **Embedding models**: The default semantic embedding model provides meaningful representations of text, but if the model cannot be loaded the system falls back to TF‑IDF.  TF‑IDF captures only term frequency and may miss synonyms or paraphrases.
* **Extractive answers**: The answer generator concatenates and paraphrases snippets from the retrieved chunks.  It does not use a large language model.  More advanced generative models could improve fluency and coherence.
* **Local storage**: All data is stored in memory and the local file system.  The current implementation does not persist the vector store across sessions.
* **Not for critical decisions**: The application is intended for educational purposes.  It should not be used for medical, legal or financial decisions without expert review.

## Future Improvements

* Persist vector stores to disk using a dedicated library such as ChromaDB or FAISS.
* Implement OCR to handle scanned documents.
* Integrate a small language model to improve answer generation while staying source‑grounded.
* Enhance the UI with better formatting and file management.
* Persist vector stores to disk using a dedicated library such as ChromaDB or FAISS.
* Implement OCR to handle scanned documents.
* Integrate a small language model to improve answer generation while staying source‑grounded.
* Enhance the UI with better formatting and file management.

## Safety and Hallucination Disclaimer

This project aims to reduce hallucination by grounding all answers in the provided documents.  Nevertheless, the extractive summarisation approach may still omit context or misinterpret information.  Always verify answers against the original sources, especially when the information is critical.  The authors are not responsible for any actions taken based on the output of this software.

## Author

**Erfan Moshfeghi** – [@Erfan-Moshfeghi](https://github.com/Erfan-Moshfeghi)

This repository is part of a portfolio of AI/NLP projects.  Contributions and feedback are welcome.