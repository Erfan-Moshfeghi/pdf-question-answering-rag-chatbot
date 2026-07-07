## Data Directory

This directory contains placeholders for different categories of data used by the project.  Nothing here is tracked by version control except for the `.gitkeep` files.  You can create sub‑folders and store your own PDFs or vector databases locally when running the application.  The sub‑directories are:

* `sample_pdfs/` – Place a few example PDF files here while developing or demonstrating the app.  This folder is empty by default.
* `uploads/` – When running the Streamlit app, user‑uploaded PDFs can be temporarily written here.  The contents of this folder are ignored by Git.
* `vector_store/` – The vector store implementation can optionally persist its database files here.  It is also ignored by Git.

Please do **not** commit copyrighted or sensitive documents into this repository.  They should remain local to your machine.