
"""Valutazione di Keyword, Vector e Hybrid Search.

Esempi:
    uv run python -m uap_project.evaluation.evaluate_retrieval
    uv run python -m uap_project.evaluation.evaluate_retrieval --top-k 10
"""

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from uap_project.retrieval.azure_retriever import (
    AzureAISearchRetriever,
)


DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_DIR = Path(__file__).with_name("results")


def normalize_section(value: str | None) -> str:
    """Normalizza titoli Markdown per il confronto."""
    text = str(value or "")
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text)

    return " ".join(text.split()).strip().casefold()


def section_key(
    source_file: str | None,
    section: str | None,
) -> tuple[str, str]:
    """Normalizza percorso e titolo della sezione."""
    filename = str(source_file or "").replace("\\", "/")
    filename = filename.rsplit("/", 1)[-1].casefold()

    return filename, normalize_section(section)


def calculate_metrics(
    documents: list,
    expected_sections: list[dict],
    top_k: int,
) -> dict:
    """Calcola metriche rispetto alle sezioni attese."""

    expected = {
        section_key(item["source_file"], item["section"])
        for item in expected_sections
    }

    retrieved = [
        section_key(
            doc.metadata.get("source_file"),
            doc.metadata.get("section"),
        )
        for doc in documents[:top_k]
    ]

    found = expected.intersection(retrieved)

    first_rank = next(
        (
            rank
            for rank, key in enumerate(retrieved, start=1)
            if key in expected
        ),
        None,
    )

    metrics = {
        "hit@1": int(first_rank == 1),
        "hit@3": int(
            first_rank is not None and first_rank <= 3
        ),
        f"hit@{top_k}": int(first_rank is not None),
        f"recall@{top_k}": len(found) / len(expected),
        f"mrr@{top_k}": (
            1 / first_rank if first_rank else 0.0
        ),
        f"full_coverage@{top_k}": int(found == expected),
        "found_sections": len(found),
        "expected_sections": len(expected),
    }

    return metrics


def create_retriever() -> AzureAISearchRetriever:
    """Crea il retriever usando le variabili d'ambiente."""

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    return AzureAISearchRetriever(
        azure_vector_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_vector_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        embeddings=embeddings,
    )


def document_info(doc, rank: int) -> dict:
    """Serializza i metadati utili al debug."""
    return {
        "rank": rank,
        "source_file": doc.metadata.get("source_file"),
        "section": doc.metadata.get("section"),
        "chunk_index": doc.metadata.get("chunk_index"),
        "search_score": doc.metadata.get("score"),
    }


def print_summary(
    results: dict[str, list[dict]],
    top_k: int,
) -> None:
    """Mostra i valori medi per metodo."""

    metric_names = [
        "hit@1",
        "hit@3",
        f"hit@{top_k}",
        f"recall@{top_k}",
        f"mrr@{top_k}",
        f"full_coverage@{top_k}",
    ]

    print("\n" + "=" * 70)
    print(f"RIEPILOGO RETRIEVAL — TOP_K={top_k}")
    print("=" * 70)

    for method, rows in results.items():
        if not rows:
            continue

        print(f"\n{method.upper()} — {len(rows)} domande")

        for metric in dict.fromkeys(metric_names):
            value = mean(
                row["metrics"][metric]
                for row in rows
            )
            print(f"  {metric:22s}: {value:.3f}")

        latency = mean(row["latency_s"] for row in rows)
        print(f"  {'latency_s':22s}: {latency:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Numero di documenti recuperati (default: 5)",
    )

    parser.add_argument(
        "--label",
        default="run",
        help="Etichetta del file di output (es. baseline, v2)",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k deve essere almeno 1")

    load_dotenv()

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    retriever = create_retriever()

    search_methods = {
        "keyword": retriever.keyword_search,
        "vector": retriever.vector_search,
        "hybrid": retriever.hybrid_search,
    }

    results = {
        name: []
        for name in search_methods
    }

    for sample in dataset:

        # Le domande non answerable richiedono
        # una valutazione separata dell'astensione.
        if not sample["answerable"]:
            continue

        expected = sample["expected_sections"]

        if not expected:
            raise ValueError(
                f"{sample['id']} non contiene expected_sections"
            )

        print(
            f"\n{sample['id']} [{sample['type']}] "
            f"{sample['user_input']}"
        )

        for method_name, search in search_methods.items():
            start = perf_counter()

            documents = list(
                search(
                    query=sample["user_input"],
                    top_k=args.top_k,
                )
            )

            latency = perf_counter() - start

            metrics = calculate_metrics(
                documents=documents,
                expected_sections=expected,
                top_k=args.top_k,
            )

            row = {
                "id": sample["id"],
                "question": sample["user_input"],
                "type": sample["type"],
                "metrics": metrics,
                "latency_s": latency,
                "documents": [
                    document_info(doc, rank)
                    for rank, doc in enumerate(documents, 1)
                ],
            }

            results[method_name].append(row)

            coverage = metrics[f"full_coverage@{args.top_k}"]

            status = "OK" if coverage else "DA RIVEDERE"

            print(
                f"  {method_name:8s} {status:11s} "
                f"Hit@{args.top_k}="
                f"{metrics[f'hit@{args.top_k}']} "
                f"Recall@{args.top_k}="
                f"{metrics[f'recall@{args.top_k}']:.2f} "
                f"MRR@{args.top_k}="
                f"{metrics[f'mrr@{args.top_k}']:.2f}"
            )

            if not coverage:
                for rank, doc in enumerate(documents, 1):
                    print(
                        f"    {rank:2d}. "
                        f"{doc.metadata.get('source_file')} | "
                        f"{doc.metadata.get('section')}"
                    )

    print_summary(results, args.top_k)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / (
        f"retrieval_{args.label}_top_{args.top_k}.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
