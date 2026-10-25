from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from uap_project.retrieval.azure_retriever import(
    AzureAISearchRetriever
)

class UAPRAGChain: 
    def __init__(
        self, 
        retriever: AzureAISearchRetriever,
        llm: AzureAISearchRetriever,
        top_k: int = 5
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
            Sei un assistente specializzato in UAP.
            
            Rispondi alle domande utilizzando esclusivamente le informazioni contenute nel contesto fornito.
            
            Regole:
            - Non inventare informazioni.
            - Non aggiungere dettagli che non siano supportati
                esplicitamente dai documenti.
            - Se il contesto non è sufficiente per rispondere,
                dichiaralo chiaramente.
            - Rispondi in italiano.
            - Cita le fonti utilizzate con [1], [2], ecc.
            - Utilizza soltanto le fonti realmente necessarie.
            - Non considerare verificata un'affermazione
                soltanto perché compare nel contesto.
            
            Prima di rispondere:
            - Verifica che il contesto contenga informazioni
            direttamente pertinenti alla domanda.
            - Non sostituire l'argomento richiesto con un argomento
            correlato ma differente.
            - Se il contesto non permette di rispondere alla domanda
            specifica, dichiara che le informazioni disponibili
            non sono sufficienti.
            - Non costruire classificazioni o categorie non
            esplicitamente supportate dal contesto.
            - Cita soltanto le fonti che supportano le affermazioni.
            """),
            (
                "human",
                """
                Contesto:
                {context}
                
                Domanda:
                {question}
                """
            )
        ])
        self.chain = (
            self.prompt
            | llm
            | StrOutputParser()
        )
    
    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        if not documents:
            return "Nessun documento recuperato"
        
        formatted_documents = list()
        
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            formatted_documents.append(
                f"""
                [{index}]
                Fonte: {metadata.get('source_file', '')}
                Sezione: {metadata.get('section', '')}

                Contenuto:
                {doc.page_content}
                """
            )
        return "\n\n".join(formatted_documents)
    
    def ask(self, question: str) -> dict:
        # Retrieval tramite Azure SDK
        documents = self.retriever.hybrid_search(
            query=question,
            top_k=self.top_k
        )

        if not documents:
            return {
                "answer": (
                    "Non ho trovato informazioni pertinenti "
                    "nei documenti disponibili."
                ),
                "sources": []
            }

        # Costruzione del contesto
        context = self._format_context(documents)

        # Generazione con LangChain
        answer = self.chain.invoke({
            "question": question,
            "context": context
        })

        # Fonti recuperate
        sources = [
            {
                "reference": index,
                "source_file": doc.metadata.get("source_file"),
                "section": doc.metadata.get("section"),
            }
            for index, doc in enumerate(documents, start=1)
        ]

        return {
            "answer": answer,
            "sources": sources
        }