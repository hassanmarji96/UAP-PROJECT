import uuid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from types import SimpleNamespace
from langchain_core.embeddings import Embeddings
from langchain_core.documents import (Document)

class AzureAISearchVectorStore:
    def __init__(
        self,
        azure_vector_endpoint,
        azure_vector_api_key,
        azure_index_name
    ) -> None:
        self.client = SearchClient(
            endpoint=azure_vector_endpoint,
            index_name=azure_index_name,
            credential=AzureKeyCredential(azure_vector_api_key)
        )
    
    def upload_chunks_in_vector_store(
        self,
        chunks: list[Document],
        embedding: Embeddings = None,
        source_file: str = None
    ) -> None:
        """_summary_ Carico i chunk nel servizio Azure AI Search

        Args:
            chunks (list[Document]): chunk
            embeddings (Embeddings): modello embedded

            source_file (str): file sorgente
        """
        documents = list()
        for chunk in chunks:
            vector = embedding.embed_query(chunk.page_content)
            documents.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_file": source_file,
                    "content": chunk.page_content,
                    "content_vector": vector
                }
            )
        result = self.client.upload_documents(documents=documents)
        uploaded = sum(item.succeeded for item in result)
        
        print(f"Chunks caricati: {uploaded}/{len(chunks)}")