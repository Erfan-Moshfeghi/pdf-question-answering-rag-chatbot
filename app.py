"""
Streamlit application for the PDF Question Answering RAG Chatbot.

Run this script with ``streamlit run app.py`` to launch a web
interface where users can upload PDF files, configure chunking
parameters, process the documents and ask questions about their
contents.  Results include the answer and supporting sources with
page numbers.
"""

from __future__ import annotations

import os
import uuid
from typing import List

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.config import DEFAULTS


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:  # type: ignore
    """Save uploaded files to the uploads directory and return their paths."""
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    paths: List[str] = []
    for uploaded_file in uploaded_files:
        # Generate a unique name to avoid collisions
        filename = uploaded_file.name
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        path = os.path.join(upload_dir, unique_name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        paths.append(path)
    return paths


def main() -> None:
    st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
    st.title("📄🔍 PDF Question Answering RAG Chatbot")
    st.markdown(
        "Upload one or more PDF files, process them into a searchable index and ask questions about their content."
    )

    # Sidebar configuration
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.number_input(
        "Chunk size (characters)",
        min_value=100,
        max_value=5000,
        value=DEFAULTS.chunk_size,
        step=50,
    )
    chunk_overlap = st.sidebar.number_input(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(DEFAULTS.chunk_overlap, int(chunk_size) - 1),
        step=10,
    )
    top_k = st.sidebar.number_input(
        "Top‑k results", min_value=1, max_value=10, value=DEFAULTS.top_k, step=1
    )
    if st.sidebar.button("Clear session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    # Initialize session state
    if "pipeline" not in st.session_state:
        st.session_state.pipeline: RAGPipeline | None = None
        st.session_state.loaded = False
        st.session_state.last_load_errors: List[str] = []

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before processing.")
        else:
            # Save uploaded files and create a new pipeline with the selected parameters
            paths = save_uploaded_files(uploaded_files)
            pipeline = RAGPipeline(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                top_k=int(top_k),
            )
            load_result = pipeline.load_documents(paths)
            st.session_state.pipeline = pipeline
            st.session_state.loaded = load_result.get("success", False)
            st.session_state.last_load_errors = load_result.get("errors", [])
            if st.session_state.loaded:
                st.success(
                    f"Processed {load_result['num_pages']} pages into {load_result['num_chunks']} chunks."
                )
            else:
                st.error("Failed to process documents. See errors below.")

    # Display errors if any
    if st.session_state.get("last_load_errors"):
        st.error("\n".join(st.session_state.last_load_errors))

    # Question input and answer display
    if st.session_state.get("loaded"):
        question = st.text_input(
            "Ask a question about the uploaded documents", value="", placeholder="Type your question here"
        )
        if st.button("Submit Question"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                pipeline: RAGPipeline = st.session_state.pipeline  # type: ignore
                result = pipeline.ask(question, top_k=int(top_k))
                if not result.get("success"):
                    st.error(result.get("error"))
                else:
                    st.subheader("Answer")
                    st.write(result["answer"])
                    sources = result.get("sources", [])
                    if sources:
                        st.subheader("Sources")
                        for src in sources:
                            doc = src.get("document")
                            page = src.get("page_number")
                            chunk_id = src.get("chunk_id")
                            score = src.get("score")
                            st.write(
                                f"**{doc}**, page {page}, chunk {chunk_id} (score {score:.3f})"
                            )
                            st.code(src.get("text"), language="text")


if __name__ == "__main__":  # pragma: no cover
    main()