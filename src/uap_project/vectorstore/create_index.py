from dotenv import load_dotenv
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

load_dotenv()

def create_index(
    vector_store_endpoint: str,
    vector_store_api_key: str,
    vector_index_name: str,
    embedding_dimensions: int = 1536
) -> None:
    """_summary_ Crea un indice vettoriale su Azure

    Args:
        endpoint (str): Azure AI Search endpoint.
        api_key (str): Azure AI Search API key.
        index_name (str): Index name.
        embedding_dimensions (str): Embedding dimensions.
    """
    if not all([
        vector_store_endpoint,
        vector_store_api_key,
        vector_index_name
    ]):
        raise ValueError("Tutti i valori del DB vettoriale sul cloud Azure, devono essere forniti!")
    
    client = SearchIndexClient(
        endpoint=vector_store_endpoint,
        credential=AzureKeyCredential(vector_store_api_key)
    )
    # Definisco i campi e metadati del mio store vettoriale
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SimpleField(
            name="document_id",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SearchableField(
            name="section",
            type=SearchFieldDataType.String,
            filterable=True,
            analyzer_name="it.microsoft"
        ),
        SearchableField(
        name="category",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
        analyzer_name="it.microsoft"
        ),
        SimpleField(
            name="content_type",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="it.microsoft"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="vector-profile"
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            filterable=False
        ),
    ]
    
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )
    
    uap_index = SearchIndex(
        name=vector_index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    client.create_or_update_index(index=uap_index)
    print(f"Indice vettoriale '{vector_index_name}' creato con successo")
    
    