import hashlib
import json
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
    
    @staticmethod
    def _embedding_text(chunk: Document) -> str:
        """
        Testo usato SOLO per calcolare l'embedding: antepone al contenuto
        il titolo del documento e la sezione, così il vettore "sa" dove
        si trova il chunk. Il campo `content` salvato resta invariato.
        """
        title = chunk.metadata.get("document_title", "")
        section = chunk.metadata.get("section", "")

        header = " > ".join(part for part in (title, section) if part)

        if not header:
            return chunk.page_content

        return f"{header}\n\n{chunk.page_content}"
    
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

        # Tutti i chunk di una chiamata appartengono allo stesso file
        # (la pipeline processa un PDF alla volta): calcolo l'ID del
        # documento una volta e cancello i suoi vecchi chunk PRIMA
        # di caricare i nuovi.
        document_id = hashlib.sha256(
            chunks[0].metadata["filename"].encode()
        ).hexdigest()
        self.delete_document_chunks(document_id=document_id)

        total_uploaded = 0

        for start in range(0, len(chunks), batch_size):

            batch = chunks[start:start + batch_size]

            # Embeddings dei documenti in batch
            texts = [self._embedding_text(chunk) for chunk in batch]

            vectors = embedding.embed_documents(texts)

            if len(vectors) != len(batch):
                raise ValueError(
                    "Numero di embeddings diverso dai chunk"
                )

            documents = []

            for chunk, vector in zip(batch, vectors):

                metadata = chunk.metadata

                source = metadata.get("source", "")

                chunk_index = metadata.get("chunk_index", 0)

                chunk_id = hashlib.sha256(
                    f"{document_id}:{chunk_index}".encode("utf-8")
                ).hexdigest()

                documents.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "source_file": metadata.get("filename", source),
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
                    "metadata": json.dumps({
                        "document_id": document_id,
                        "source_file": metadata.get("filename", source),
                        "section": metadata.get("section", ""),
                        "category": metadata.get("category", "uncategorized"),
                        "content_type": metadata.get("content_type", "text"),
                        "chunk_index": chunk_index,
                    }, ensure_ascii=False),
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
        
    def delete_document_chunks(self, document_id: str) -> None:
        """Cancella tutti i chunk di un documento prima di reindicizzarlo."""
        results = self.client.search(
            search_text="*",
            filter=f"document_id eq '{document_id}'",
            select=["id"],
            top=1000,
        )
        keys = [{"id": r["id"]} for r in results]
        if keys:
            self.client.delete_documents(documents=keys)