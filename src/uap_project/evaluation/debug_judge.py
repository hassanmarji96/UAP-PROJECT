"""Diagnostica dei giudici Ragas, un pezzo alla volta e con timeout brevi.

uv run python -u -m uap_project.evaluation.debug_judge
"""
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

logging.basicConfig(level=logging.INFO)


def step(label, fn) -> None:
    start = time.perf_counter()
    try:
        result = fn()
        print(f"OK   {label}: {result!r} ({time.perf_counter() - start:.1f}s)")
    except Exception as exc:
        print(
            f"FAIL {label}: {type(exc).__name__}: {exc} "
            f"({time.perf_counter() - start:.1f}s)"
        )


def main() -> None:
    load_dotenv()
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    chat = AzureChatOpenAI(
        azure_deployment=os.getenv(
            "AZURE_OPENAI_EVAL_DEPLOYMENT",
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        ),
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=60,
        max_retries=1,
    )
    emb = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    step("chat diretta", lambda: chat.invoke("Rispondi solo: OK").content)
    step(
        "chat con temperature",
        lambda: chat.invoke("Rispondi solo: OK", temperature=0.3).content,
    )
    step(
        "chat con n=3",
        lambda: len(
            chat.generate([[HumanMessage("Rispondi solo: OK")]], n=3)
            .generations[0]
        ),
    )
    step("embeddings", lambda: len(emb.embed_query("test")))

    run_config = RunConfig(
        timeout=60, max_retries=2, max_wait=10, log_tenacity=True
    )
    
    judge_llm = LangchainLLMWrapper(
        chat, run_config=run_config, bypass_temperature=True
    )
    judge_emb = LangchainEmbeddingsWrapper(emb, run_config=run_config)

    sample = SingleTurnSample(
        user_input="Qual è la capitale d'Italia?",
        response="La capitale d'Italia è Roma.",
        retrieved_contexts=["Roma è la capitale d'Italia."],
        reference="Roma",
    )
    step(
        "ragas faithfulness",
        lambda: Faithfulness(llm=judge_llm).single_turn_score(sample),
    )
    step(
        "ragas answer_relevancy",
        lambda: AnswerRelevancy(
            llm=judge_llm, embeddings=judge_emb
        ).single_turn_score(sample),
    )


if __name__ == "__main__":
    main()