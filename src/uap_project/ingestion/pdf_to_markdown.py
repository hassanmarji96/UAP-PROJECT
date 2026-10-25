from pathlib import Path
import pymupdf4llm

class PDFToMarkDownConverter:
    """_summary_: Questa classe si occupa di convertire i file passati nella ingestion dal formato PDF, a markdown.
    Ciò viene fatto in quanto i file MD, preservano la struttura, di conseguenza, facilita ed aiuta la fase di chunking.
    """
    
    def convert_pdf(self, pdf_path: str, output_directory: str) -> str:
        """_summary_
        Converte un documento PDF in un file markdown strutturato
        Args:
            pdf_path (str): cartella sorgente in cui è contenuto il file
            output_directory (str): cartella di destinazione

        Returns:
            Markdown file generato
        """
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"Il file {pdf_file.name} non esiset nel percorso specificato: {pdf_path}!")
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        markdown_content = pymupdf4llm.to_markdown(str(pdf_file))
        markdown_file = output_path / f"{pdf_file.stem}.md"
        
        markdown_file.write_text(markdown_content, encoding="utf-8")
        return str(markdown_file)
    
    def convert_all_files_in_directory(self, input_directory: str, output_directory: str) -> list[str]:
        """_summary_ Funzione principale che si appoggia alla funzione `convert_pdf` che itera su tutti
        i file presenti nella cartella e li converte in markdown.

        Args:
            input_directory (str): cartella sorgente in cui è contenuto il file
            output_directory (str): cartella di destinazione

        Returns:
            list[str]: Lista generata dei file MD
        """
        
        input_path = Path(input_directory)
        markdown_files = list()
        
        for idx, pdf_file in enumerate(input_path.glob("*.pdf")):
            print(f"Elaborando il file #{idx + 1}, con nome: {pdf_file.name}")
            markdown_file = self.convert_pdf(
                pdf_path=pdf_file,
                output_directory=output_directory
            )
            markdown_files.append(markdown_file)
        
        return markdown_files
   