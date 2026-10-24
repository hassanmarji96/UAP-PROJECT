import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from uap_project.pipeline import run_pipeline
from uap_project.vectorstore.index_documents import (
    AzureAISearchVectorStore
)
from uap_project.vectorstore.create_index import (
    create_index
)

def main() -> None:

    load_dotenv()

    project_directory = Path(__file__).resolve().parent

    pdf_directory = project_directory / "docs" / "raw_pdf"
    markdown_directory = project_directory / "docs" / "markdown"

    # Embedding Model
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    )

    # Configurazione Vector Store
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_API_KEY"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    # Crea o aggiorna lo schema dell'indice
    create_index(
        vector_store_endpoint=endpoint,
        vector_store_api_key=api_key,
        vector_index_name=index_name,
        embedding_dimensions=1536
    )

    vector_store = AzureAISearchVectorStore(
        azure_vector_endpoint=endpoint,
        azure_vector_api_key=api_key,
        azure_index_name=index_name
    )

    # Individua i PDF
    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        print("Nessun PDF trovato.")
        return

    # Avvia la pipeline
    for pdf_file in pdf_files:

        print(f"Elaborazione: {pdf_file.name}")

        run_pipeline(
            pdf_path=pdf_file,
            markdown_directory=markdown_directory,
            embedding=embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":
    main()