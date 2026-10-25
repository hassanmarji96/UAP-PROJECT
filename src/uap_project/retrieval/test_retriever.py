import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from uap_project.retrieval.azure_retriever import (
    AzureAISearchRetriever
)

load_dotenv()

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

azure_retriever = AzureAISearchRetriever(
    azure_vector_endpoint=endpoint,
    azure_vector_api_key=api_key,
    azure_index_name=index_name,
    embeddings=embeddings
)

print("Per favore inserisci la tua domanda:")
query = input()


results = {
    "KEYWORD": azure_retriever.keyword_search(query),
    "VECTOR": azure_retriever.vector_search(query),
    "HYBRID": azure_retriever.hybrid_search(query),
}

for search_type, documents in results.items():

    print(f"\n{'=' * 60}")
    print(f"RICERCA: {search_type}")
    print(f"{'=' * 60}")

    for i, document in enumerate(documents, start=1):

        print(f"\n--- DOCUMENTO {i} ---")

        print(
            "Fonte:",
            document.metadata.get("source_file")
        )

        print(
            "Sezione:",
            document.metadata.get("section")
        )

        print(
            "Categoria:",
            document.metadata.get("category")
        )

        print(
            "Score:",
            document.metadata.get("score")
        )
