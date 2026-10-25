
"""Valutazione deterministica del retrieval di UAP-PROJECT.

Esegui dalla root del progetto:
    uv run python -m uap_project.evaluation.evaluate_retrieval
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from uap_project.retrieval.azure_retriever import AzureAISearchRetriever


DATASET_PATH = Path(__file__).with_name("dataset.json")
TOP_K = 10


def normalize_section(value: str | None) -> str:
    """Normalizza il titolo della sezione per confrontarlo con i metadata."""

    text = value or ""
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text)

    return " ".join(text.split()).strip().casefold()


def section_key(
    filename: str | None,
    section: str | None
) -> tuple[str, str]:
    """Restituisce una chiave normalizzata (filename, section)."""

    # Azure restituisce il percorso completo,
    # mentre il dataset usa soltanto il nome del file.
    normalized_path = (filename or "").replace("\\", "/")

    return (
        normalized_path.rsplit("/", 1)[-1].casefold(),
        normalize_section(section)
    )


def score_results(
    documents: list,
    expected_sections: list[dict],
    top_k: int
) -> dict:
    """Calcola le metriche di retrieval confrontando le sezioni."""

    # Sezioni che ci aspettiamo di recuperare
    expected = {
        section_key(item["source_file"], item["section"])
        for item in expected_sections
    }

    # Sezioni effettivamente recuperate
    retrieved = [
        section_key(
            doc.metadata.get("source_file"),
            doc.metadata.get("section")
        )
        for doc in documents[:top_k]
    ]

    # Prima posizione in cui compare una sezione pertinente
    first_rank = next(
        (
            i
            for i, key in enumerate(retrieved, 1)
            if key in expected
        ),
        None
    )

    # Intersezione tra sezioni attese e recuperate
    found = expected.intersection(retrieved)

    return {
        "hit@1": int(first_rank == 1),
        "hit@3": int(first_rank is not None and first_rank <= 3),
        "hit@5": int(first_rank is not None and first_rank <= 5),
        "recall@5": len(found) / len(expected),
        "mrr@5": (
            1 / first_rank
            if first_rank is not None
            else 0.0
        ),
        "found_sections": len(found),
        "expected_sections": len(expected),
    }


def main() -> None:

    # 1. Caricamento configurazione
    load_dotenv()

    # 2. Caricamento dataset
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    # 3. Configurazione Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    # 4. Inizializzazione del retriever
    retriever = AzureAISearchRetriever(
        azure_vector_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_vector_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        embeddings=embeddings,
    )

    # 5. Metodi di ricerca da confrontare
    search_methods = {
        "keyword": retriever.keyword_search,
        "vector": retriever.vector_search,
        "hybrid": retriever.hybrid_search,
    }

    metrics = (
        "hit@1",
        "hit@3",
        "hit@5",
        "recall@5",
        "mrr@5"
    )

    rows = {
        name: []
        for name in search_methods
    }

    # 6. Esecuzione dei test
    for sample in dataset:

        # Le domande senza risposta verranno valutate
        # successivamente nella fase di Generation.
        if not sample["answerable"]:
            continue

        expected_sections = sample["expected_sections"]

        if not expected_sections:
            raise ValueError(
                f"Mancano expected_sections per {sample['id']}"
            )

        print(
            f"\n{sample['id']} "
            f"[{sample['type']}] "
            f"{sample['user_input']}"
        )

        # Eseguiamo i tre metodi di retrieval
        for name, search in search_methods.items():

            documents = list(
                search(
                    query=sample["user_input"],
                    top_k=TOP_K
                )
            )

            # Calcolo delle metriche
            result = score_results(
                documents,
                expected_sections,
                TOP_K
            )

            rows[name].append(result)

            status = (
                "OK"
                if result["recall@5"] == 1
                else "DA RIVEDERE"
            )

            print(
                f"  {name:8s} {status:11s} "
                f"Hit@5={result['hit@5']}  "
                f"Recall@5={result['recall@5']:.2f}  "
                f"MRR@5={result['mrr@5']:.2f}"
            )

            # Mostra i risultati quando il retrieval
            # non trova tutte le sezioni attese.
            if status != "OK":

                for position, doc in enumerate(
                    documents, 1
                ):
                    print(
                        f"     {position}. "
                        f"{doc.metadata.get('source_file')} | "
                        f"{doc.metadata.get('section')}"
                    )

    # 7. Riepilogo finale
    print("\n" + "=" * 72)
    print(
        "RISULTATI COMPLESSIVI "
        "(media sulle sole domande con risposta)"
    )
    print("=" * 72)

    for name, results in rows.items():

        if not results:
            continue

        summary = "  ".join(
            f"{metric}="
            f"{sum(row[metric] for row in results) / len(results):.3f}"
            for metric in metrics
        )

        print(
            f"{name:8s} | "
            f"n={len(results):2d} | "
            f"{summary}"
        )


if __name__ == "__main__":
    main()
