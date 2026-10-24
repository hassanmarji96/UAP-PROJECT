import hashlib

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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
        embedding: Embeddings,
        batch_size: int = 100,
    ) -> None:
        """
        Genera gli embeddings e indicizza i chunk
        su Azure AI Search, preservando i metadata.
        """

        if not chunks:
            print("Nessun chunk da indicizzare.")
            return

        if batch_size <= 0:
            raise ValueError("batch_size deve essere > 0")

        total_uploaded = 0

        for start in range(0, len(chunks), batch_size):

            batch = chunks[start:start + batch_size]

            # Embeddings dei documenti in batch
            texts = [chunk.page_content for chunk in batch]

            vectors = embedding.embed_documents(texts)

            if len(vectors) != len(batch):
                raise ValueError(
                    "Numero di embeddings diverso dai chunk"
                )

            documents = []

            for chunk, vector in zip(batch, vectors):

                metadata = chunk.metadata

                source = metadata.get("source", "")

                document_id = hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()
                chunk_index = metadata.get("chunk_index", 0)

                chunk_id = hashlib.sha256(
                    f"{document_id}:{chunk_index}".encode("utf-8")
                ).hexdigest()

                documents.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "source_file": source,
                    "section": metadata.get("section", ""),
                    "category": metadata.get(
                        "category", "uncategorized"
                    ),
                    "content_type": metadata.get(
                        "content_type", "text"
                    ),
                    "chunk_index": chunk_index,
                    "content": chunk.page_content,
                    "content_vector": vector,
                })

            results = self.client.upload_documents(
                documents=documents
            )

            failures = [
                result for result in results
                if not result.succeeded
            ]

            total_uploaded += (
                len(results) - len(failures)
            )

            if failures:
                errors = [
                    f"{r.key}: {r.error_message}"
                    for r in failures
                ]
                raise RuntimeError(
                    "Errore indicizzazione: "
                    + "; ".join(errors)
                )

        print(
            f"Chunk caricati: "
            f"{total_uploaded}/{len(chunks)}"
        )