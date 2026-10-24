import os
from pathlib import Path
from uap_project.pipeline import run_pipeline

def main() -> None:
    print("Preparo la conversione dei file")
    files_to_ingest_root = Path(__file__).resolve().parents[0]
    pdf_directory = files_to_ingest_root/ "docs"/"raw_pdf"
    output_directory = files_to_ingest_root/ "docs"/"markdown"
    
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        print(f"Nessun file pdf trovato nella cartella {files_to_ingest_root}")
        return
    
    for pdf_file in pdf_files:
        print(f"PDF in elaborazione... {pdf_file.name}")
        run_pipeline(
            pdf_path=str(pdf_file),
            markdown_directory=output_directory
        )

if __name__ == "__main__":
    main()