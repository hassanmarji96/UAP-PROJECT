from pathlib import Path

from uap_project.ingestion.pdf_to_markdown import (
    PDFToMarkDownConverter
)
from uap_project.ingestion.structured_aware_chunking import (
    chunk_markdown_directory,
    chunk_markdown
)


def run_pipeline(
    pdf_path: str,
    markdown_directory: str,
    embeddings = None,
    vector_store = None
) -> dict:
    
    pdf_file = Path(pdf_path)
    # Step 1: converto i documenti per poterli chunkare
    pdf_to_markdown_converter = PDFToMarkDownConverter()
    markdown_files = pdf_to_markdown_converter.convert_pdf(
        pdf_path=str(pdf_file),
        output_directory=markdown_directory
    )
    # Step 2: creo i chunk e li carico nell'indice del vector-store
