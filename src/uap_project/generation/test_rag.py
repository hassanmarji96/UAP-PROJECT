import os

from dotenv import load_dotenv
from langchain_openai import (
    AzureOpenAIEmbeddings,
    AzureChatOpenAI
)

from uap_project.retrieval.azure_retriever import (
    AzureAISearchRetriever
)
from uap_project.generation.rag_chain import (
    UAPRAGChain
)

def main():

    load_dotenv()

    embeddings = AzureOpenAIEmbeddings(
           azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
           azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
           api_key=os.environ["AZURE_OPENAI_API_KEY"],
           api_version=os.environ["AZURE_OPENAI_API_VERSION"]
       )
    
    retriever = AzureAISearchRetriever(
        azure_vector_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_vector_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        embeddings=embeddings
    )

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    )

    rag = UAPRAGChain(
        retriever=retriever,
        llm=llm,
        top_k=5
    )

    while True:

        question = input(
            "\nInserisci la domanda (exit per uscire): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = rag.ask(question)

        print("\nRISPOSTA:")
        print(result["answer"])

        print("\nFONTI RECUPERATE:")

        for source in result["sources"]:
            print(
                f"[{source['reference']}] "
                f"{source['source_file']} "
                f"- {source['section']}"
            )


if __name__ == "__main__":
    main()