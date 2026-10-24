from pathlib import Path
from langchain_core.documents import Document
from uap_project.ingestion.pdf_to_markdown import (
    PDFToMarkDownConverter
)
from uap_project.ingestion.structured_aware_chunking import (
    chunk_markdown_directory,
    chunk_markdown
)
from uap_project.vectorstore.create_index import create_index

def run_pipeline(
    pdf_path: str | Path,
    markdown_directory: str | Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> list[Document]:
    
    pdf_file = Path(pdf_path)
    # Step 1: converto i documenti per poterli chunkare
    pdf_to_markdown_converter = PDFToMarkDownConverter()
    markdown_files = pdf_to_markdown_converter.convert_pdf(
        pdf_path=str(pdf_file),
        output_directory=markdown_directory
    )
    # Step 2: creo i chunk e li carico nell'indice del vector-store
    chunks = list()
    for markdown_file in markdown_files:
        document_chunk = chunk_markdown(
            file_path=markdown_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks.extend(document_chunk)
        