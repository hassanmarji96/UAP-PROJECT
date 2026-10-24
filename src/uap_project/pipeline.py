from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from uap_project.ingestion.pdf_to_markdown import (
    PDFToMarkDownConverter
)
from uap_project.ingestion.structured_aware_chunking import (
    chunk_markdown
)
from uap_project.vectorstore.index_documents import (
    AzureAISearchVectorStore
)

def run_pipeline(
    pdf_path: str | Path,
    markdown_directory: str | Path,
    embedding: Embeddings,
    vector_store: AzureAISearchVectorStore,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:

    # Step 1: conversione PDF -> Markdown
    pdf_to_markdown_converter = PDFToMarkDownConverter()

    markdown_file = pdf_to_markdown_converter.convert_pdf(
        pdf_path=str(pdf_path),
        output_directory=str(markdown_directory)
    )

    # Step 2: chunking del Markdown generato
    chunks = chunk_markdown(
        file_path=markdown_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"Generati {len(chunks)} chunk")

    # Step 3: embeddings + indicizzazione
    vector_store.upload_chunks_in_vector_store(
        chunks=chunks,
        embedding=embedding
    )

    return chunks
        