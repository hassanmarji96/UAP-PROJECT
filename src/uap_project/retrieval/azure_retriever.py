from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class AzureAISearchRetriever:
    def __init__(
        self,
        azure_vector_endpoint: str,
        azure_vector_api_key: str,
        azure_index_name: str,
        embeddings: Embeddings
    ) -> None:
        self.client = SearchClient(
            endpoint=azure_vector_endpoint,
            index_name=azure_index_name,
            credential=AzureKeyCredential(azure_vector_api_key)
        )
        self.embeddings = embeddings
    
    def _to_documents(self, results) -> list[Document]:
        """Converte i risultati Azure in Document LangChain."""
        documents = list()
        
        for result in results:
            documents.append(
                Document(
                    page_content=result["content"],
                    metadata = {
                        "id": result["id"],
                        "source_file": result.get("source_file"),
                        "section": result.get("section"),
                        "category": result.get("category"),
                        "chunk_index": result.get("chunk_index"),
                        "score": result.get("@search.score"),
                    }
                )
            )
        return documents
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[Document]:
        
        results = self.client.search(
            search_text=query,
            top=top_k
        )
        return results
    
    def vector_search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[Document]:
        
        query_vector = self.embeddings.embed_query(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content-vector"
        )
        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )
        
        print(f"Risultati vettoriali: {results[:20]}")
        return self._to_documents(results=results)
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None
    ) -> list[Document]:
        
        query_vector = self.embeddings.embed_query(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=max(top_k, 20),
            fields="content_vector"
        )
        
        search_filter = None
        
        if category:
            safe_category = category.replace("'", "''")
            search_filter = f"category eq '{safe_category}'"
            
        results = self.client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=search_filter,
            top=top_k
        )
        
        return self._to_documents(results=results)
    
    