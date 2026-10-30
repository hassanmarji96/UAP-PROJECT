
"""Evaluation della generazione RAG con Ragas 0.4.x.

Esecuzione:
    uv run python -m uap_project.evaluation.evaluate_generation

Per testare soltanto 3 domande:
    uv run python -m uap_project.evaluation.evaluate_generation --limit 3
"""

import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    Faithfulness,
)
from ragas.run_config import RunConfig

from uap_project.retrieval.azure_retriever import (
    AzureAISearchRetriever,
)
from uap_project.generation.rag_chain import UAPRAGChain


DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_DIR = Path(__file__).with_name("results")
TOP_K = 10


def create_rag() -> UAPRAGChain:
    """Inizializza il RAG esistente."""

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    retriever = AzureAISearchRetriever(
        azure_vector_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_vector_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        embeddings=embeddings,
    )

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    return UAPRAGChain(
        retriever=retriever,
        llm=llm,
        top_k=TOP_K,
    )


def create_evaluators() -> dict:
    """Configura i modelli giudice e le metriche Ragas."""

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    judge_deployment = os.getenv(
        "AZURE_OPENAI_EVAL_DEPLOYMENT",
        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )
    
    run_config = RunConfig(timeout=90, max_retries=3, max_wait=15)

    judge_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_deployment=judge_deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        ),
        run_config=run_config,
        bypass_temperature=True,
    )
    
    judge_embeddings = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    )
    
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=judge_llm,
            embeddings=judge_embeddings
        ),
        "answer_correctness": AnswerCorrectness(
            llm=judge_llm,
            embeddings=judge_embeddings,
            answer_similarity=AnswerSimilarity(
                embeddings=judge_embeddings
            ),
        )
    }


def generate_sample(
    rag: UAPRAGChain,
    sample: dict,
) -> dict:
    """Esegue retrieval e generation conservando gli stessi chunk."""

    question = sample["user_input"]

    documents = list(
        rag.retriever.hybrid_search(
            query=question,
            top_k=rag.top_k,
        )
    )

    contexts = [doc.page_content for doc in documents]

    sources = [
        {
            "source_file": doc.metadata.get("source_file"),
            "section": doc.metadata.get("section"),
        }
        for doc in documents
    ]

    # Usiamo gli stessi documenti per la generazione
    # e per la valutazione: nessuna seconda ricerca.
    if documents:
        context = rag._format_context(documents)

        response = rag.chain.invoke({
            "question": question,
            "context": context,
        })
    else:
        response = (
            "Non ho trovato informazioni pertinenti "
            "nei documenti disponibili."
        )

    return {
        "id": sample["id"],
        "type": sample["type"],
        "user_input": question,
        "reference": sample.get("reference"),
        "answerable": sample["answerable"],
        "retrieved_contexts": contexts,
        "sources": sources,
        "response": response,
        "scores": {},
        "errors": {},
    }


def evaluate_sample(
    item: dict,
    evaluators: dict,
) -> None:
    """Calcola le metriche sulle domande con risposta."""

    if not item["answerable"]:
        # I casi senza risposta richiedono un test di
        # astensione separato, non Answer Correctness.
        return

    metric_inputs = {
        "faithfulness": {
            "user_input": item["user_input"],
            "response": item["response"],
            "retrieved_contexts": item["retrieved_contexts"],
        },
        "answer_relevancy": {
            "user_input": item["user_input"],
            "response": item["response"],
        },
        "answer_correctness": {
            "user_input": item["user_input"],
            "response": item["response"],
            "reference": item["reference"],
        },
    }

    for name, evaluator in evaluators.items():
        try:
            sample = SingleTurnSample(**metric_inputs[name])

            score = float(evaluator.single_turn_score(sample))

            if not math.isfinite(score):
                raise ValueError("Score non finito")

            item["scores"][name] = score

        except Exception as exc:
            # Non trasformiamo gli errori in zero:
            # uno score non calcolato non è uno score basso.
            item["scores"][name] = None
            item["errors"][name] = (
                f"{type(exc).__name__}: {exc}"
            )


def print_summary(results: list[dict]) -> None:
    """Stampa le medie delle metriche calcolate."""

    print("\n" + "=" * 65)
    print("RISULTATI GENERATION EVALUATION")
    print("=" * 65)

    answerable = [
        row for row in results if row["answerable"]
    ]

    for metric in (
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ):
        values = [
            row["scores"][metric]
            for row in answerable
            if row["scores"].get(metric) is not None
        ]

        if values:
            print(
                f"{metric:22s} "
                f"{mean(values):.3f} "
                f"(n={len(values)}/{len(answerable)})"
            )
        else:
            print(f"{metric:22s} N/D")

    unanswerable = [
        row for row in results if not row["answerable"]
    ]

    print(
        f"\nDomande senza risposta da verificare "
        f"manualmente: {len(unanswerable)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di domande da valutare",
    )
    args = parser.parse_args()

    load_dotenv()

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit deve essere positivo")
        dataset = dataset[:args.limit]

    rag = create_rag()
    evaluators = create_evaluators()

    results = []

    for index, sample in enumerate(dataset, start=1):
        print(
            f"\n[{index}/{len(dataset)}] "
            f"{sample['id']}: {sample['user_input']}"
        )

        try:
            item = generate_sample(rag, sample)

            evaluate_sample(item, evaluators)

            results.append(item)

            print("RISPOSTA:", item["response"])

            if item["answerable"]:
                for metric, score in item["scores"].items():
                    display = (
                        f"{score:.3f}"
                        if score is not None
                        else "ERRORE"
                    )
                    print(f"  {metric}: {display}")
            else:
                print(
                    "  Domanda senza risposta: "
                    "verificare l'astensione."
                )

        except Exception as exc:
            print(
                f"ERRORE {sample['id']}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append({
                "id": sample["id"],
                "user_input": sample["user_input"],
                "answerable": sample["answerable"],
                "response": None,
                "scores": {},
                "error": str(exc),
            })

    print_summary(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = RESULTS_DIR / (
        f"generation_eval_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
