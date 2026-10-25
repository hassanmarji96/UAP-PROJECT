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


query = (
    "Quali eventi UAP importanti sono riportati "
    "tra il 2004 e il 2022?"
)

methods = {
    "keyword": azure_retriever.keyword_search,
    "vector": azure_retriever.vector_search,
    "hybrid": azure_retriever.hybrid_search,
}

for name, search in methods.items():
    print(f"\n===== {name.upper()} =====")

    documents = search(query=query, top_k=15)

    for rank, doc in enumerate(documents, start=1):
        print(f"\n{rank}. Score: {doc.metadata.get('score')}")
        print(f"File: {doc.metadata.get('source_file')}")
        print(f"Sezione: {doc.metadata.get('section')}")
        print(f"Contenuto: {doc.page_content[:250]}")