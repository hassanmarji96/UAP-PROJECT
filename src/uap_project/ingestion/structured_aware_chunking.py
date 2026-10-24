"""Chunking di file Markdown che rispetta la struttura del documento.

Il documento viene prima diviso per intestazioni (#, ##, ###, ####), poi ogni
sezione viene suddivisa in chunk di dimensione massima configurabile.
Le tabelle Markdown non vengono mai spezzate a metà riga: vengono divise solo
tra una riga e l'altra, ripetendo l'intestazione delle colonne in ogni chunk.
Ogni chunk porta con sé metadati (fonte, titolo, sezione, indice, tipo di contenuto).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv

load_dotenv()

# Riga separatore di una tabella Markdown, es. "| --- | :---: |" (almeno 3 trattini per colonna).
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
# Intestazione Markdown (da # a ######); il gruppo 1 cattura il testo del titolo.
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)

# Forzo l'etichettamento io, siccome è un progetto studio. 
# In ambienti reali, potrei farlo assegnare automaticamente da un LLM che classifica il contenuto dei documenti
DOCUMENT_CATEGORIES = {
    "PDF_1_Primi_avvistamenti_UAP_dove_e_quando": "historical_sightings",
    "PDF_2_Come_vengono_gestiti_i_casi_UAP": "case_management",
    "PDF_3_Casistiche_UAP": "case_classification",
    "PDF_4_Fonti_ufficiali_UAP": "official_sources",
}


def _is_table_row(line: str) -> bool:
    """Restituisce True se la riga appartiene a una tabella Markdown (inizia e finisce con '|')."""
    return line.lstrip().startswith("|") and line.rstrip().endswith("|")


def _split_table(table_lines: list[str], max_chars: int) -> list[str]:
    """Divide una tabella Markdown per righe complete, ripetendo l'intestazione delle colonne.

    Args:
        table_lines: righe della tabella (intestazione, separatore e righe dati).
        max_chars: dimensione massima in caratteri di ciascun chunk.

    Returns:
        Lista di stringhe, ciascuna una tabella valida e autonoma.

    Le righe più lunghe di max_chars vengono volutamente mantenute intere.
    """
    if not table_lines:
        return []

    # Intestazione = prima riga + riga separatore (se presente); verrà ripetuta in ogni chunk.
    header = table_lines[:2] if len(table_lines) > 1 and _TABLE_SEPARATOR.match(table_lines[1]) else []
    rows = table_lines[len(header):]
    if not rows:
        return ["\n".join(table_lines)]

    prefix = "\n".join(header)
    results: list[str] = []
    current: list[str] = []

    for row in rows:
        # Si prova ad aggiungere la riga al chunk corrente: se si supera max_chars
        # si chiude il chunk e se ne apre uno nuovo (con la stessa intestazione).
        candidate = "\n".join(filter(None, [prefix, *current, row]))
        if current and len(candidate) > max_chars:
            results.append("\n".join(filter(None, [prefix, *current])))
            current = [row]
        else:
            current.append(row)

    if current:
        results.append("\n".join(filter(None, [prefix, *current])))
    return results


def _split_section_preserving_tables(
    section: Document,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[Document]:
    """Divide una sezione separando testo e tabelle.

    Il testo normale viene diviso ricorsivamente dallo splitter; le tabelle
    vengono divise solo ai confini di riga. Ogni chunk eredita i metadati della
    sezione e riceve in più `content_type` ("text" oppure "table").
    """
    lines = section.page_content.splitlines()
    # Ogni blocco è una coppia (tipo, righe): tipo = "text" | "table".
    blocks: list[tuple[str, list[str]]] = []
    prose: list[str] = []
    table: list[str] = []

    def flush_prose() -> None:
        """Chiude il blocco di testo corrente (ignorando blocchi vuoti)."""
        if prose and "\n".join(prose).strip():
            blocks.append(("text", prose.copy()))
        prose.clear()

    def flush_table() -> None:
        """Chiude il blocco di tabella corrente."""
        if table:
            blocks.append(("table", table.copy()))
        table.clear()

    # Scansione riga per riga: al passaggio testo<->tabella si chiude il blocco precedente.
    for line in lines:
        if _is_table_row(line):
            flush_prose()
            table.append(line)
        else:
            flush_table()
            prose.append(line)
    flush_table()
    flush_prose()

    chunks: list[Document] = []
    for kind, block_lines in blocks:
        content = "\n".join(block_lines).strip()
        if kind == "table":
            parts = _split_table(block_lines, chunk_size)
        else:
            parts = splitter.split_text(content)

        for part in parts:
            # Si scartano i chunk vuoti; i metadati della sezione vengono propagati.
            if part.strip():
                chunks.append(
                    Document(
                        page_content=part.strip(),
                        metadata={**section.metadata, "content_type": kind},
                    )
                )
    return chunks


def chunk_markdown(
    file_path: str | Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Divide un file Markdown in chunk (Document LangChain) con metadati.

    Args:
        file_path: percorso di un file .md esistente.
        chunk_size: dimensione massima di un chunk, in caratteri (non token).
        chunk_overlap: sovrapposizione tra chunk consecutivi di testo, in caratteri.

    Returns:
        Lista di Document con metadati: source, filename, document_title,
        intestazioni (title/section/...), content_type, chunk_index, chunk_id.

    Raises:
        ValueError: parametri non validi o file non esistente / non .md.

    Le righe di tabella molto lunghe restano intere, anche se superano chunk_size.
    """
    # Validazione: l'overlap deve essere strettamente minore della dimensione del chunk.
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Usare chunk_size > 0 e 0 <= chunk_overlap < chunk_size")

    path = Path(file_path)
    if not path.is_file() or path.suffix.lower() != ".md":
        raise ValueError(f"Atteso un file .md esistente: {path}")
    
    category = DOCUMENT_CATEGORIES.get(
        path.stem,
        "uncategorized"
    )
    # utf-8-sig rimuove l'eventuale BOM all'inizio del file.
    markdown = path.read_text(encoding="utf-8-sig")
    if not markdown.strip():
        return []

    # Titolo del documento: prima intestazione trovata (senza grassetto), altrimenti il nome file.
    title_match = _HEADER.search(markdown)
    document_title = (
        re.sub(r"\*\*|__", "", title_match.group(1)).strip()
        if title_match
        else path.stem
    )

    # Primo livello: divisione per intestazioni. strip_headers=False mantiene il titolo
    # nel testo del chunk, utile per il contesto semantico; la gerarchia finisce nei metadati.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
            ("####", "subsubsection"),
        ],
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown)
    # Secondo livello: divisione del testo lungo, preferendo paragrafi, poi righe, frasi, parole.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    output: list[Document] = []
    for section in sections:
        section.metadata.update(
            {
                "source": str(path),
                "filename": path.name,
                "document_title": document_title,
                "category": category,
            }
        )
        output.extend(
            _split_section_preserving_tables(
                section,
                recursive_splitter,
                chunk_size
            )
        )

    # Indice progressivo e ID univoco (es. "nomefile_0007") assegnati a fine elaborazione.
    for i, document in enumerate(output):
        document.metadata["chunk_index"] = i
        document.metadata["chunk_id"] = f"{path.stem}_{i:04d}"

    return output


def chunk_markdown_directory(
    directory: str | Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Applica `chunk_markdown` a tutti i file .md di una cartella (ricorsivo, in ordine di percorso).

    Args:
        directory: cartella da scansionare.
        chunk_size: dimensione massima di un chunk, in caratteri.
        chunk_overlap: sovrapposizione tra chunk, in caratteri.

    Returns:
        Lista unica di Document di tutti i file.

    Raises:
        ValueError: se il percorso non è una cartella.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Non è una cartella: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        documents.extend(
            chunk_markdown(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return documents

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Chunking di file Markdown basato sulla struttura")
#     parser.add_argument("path", type=Path, help="File Markdown o cartella")
#     parser.add_argument("--chunk-size", type=int, default=1200)
#     parser.add_argument("--chunk-overlap", type=int, default=150)
#     args = parser.parse_args()

#     if args.path.is_dir():
#         chunks = chunk_markdown_directory(
#             args.path,
#             chunk_size=args.chunk_size,
#             chunk_overlap=args.chunk_overlap,
#         )
#     else:
#         chunks = chunk_markdown(
#             args.path,
#             chunk_size=args.chunk_size,
#             chunk_overlap=args.chunk_overlap,
#         )

#     print(f"Generati {len(chunks)} chunk")
#     for chunk in chunks:
#         print(f"\n--- {chunk.metadata['chunk_id']} ---")
#         print("Metadati:", chunk.metadata)
#         print(chunk.page_content)


# if __name__ == "__main__":
#     main()