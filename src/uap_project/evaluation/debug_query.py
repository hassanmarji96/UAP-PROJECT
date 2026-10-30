"""Diagnostica: dove si piazzano i chunk attesi per una domanda.

uv run python -m uap_project.evaluation.debug_query "domanda" --top-k 20 --mark "Cronologia"
"""
import argparse

from dotenv import load_dotenv

from uap_project.evaluation.evaluate_retrieval import create_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--mark",
        default=None,
        help="Testo da evidenziare nel titolo della sezione",
    )
    args = parser.parse_args()

    load_dotenv()
    retriever = create_retriever()

    methods = {
        "keyword": retriever.keyword_search,
        "vector": retriever.vector_search,
        "hybrid": retriever.hybrid_search,
    }

    for name, search in methods.items():
        print(f"\n=== {name.upper()} ===")
        documents = search(query=args.question, top_k=args.top_k)

        for rank, doc in enumerate(documents, 1):
            meta = doc.metadata
            filename = (
                str(meta.get("source_file", ""))
                .replace("\\", "/")
                .rsplit("/", 1)[-1]
            )
            section = str(meta.get("section", ""))
            flag = (
                "★"
                if args.mark and args.mark.lower() in section.lower()
                else " "
            )
            print(
                f"{flag} {rank:2d}. {filename[:34]:34s} "
                f"#{str(meta.get('chunk_index')):>3} | "
                f"{section[:52]:52s} | {meta.get('score') or 0:.3f}"
            )


if __name__ == "__main__":
    main()