from typing import List, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str  # Question initiale de l'utilisateur (ne change jamais)
    search_query: str  # Requête optimisée pour ChromaDB (modifiée par rewrite_query)
    documents: List[Document]  # Les chunks ChromaDB
    generation: str
    retries: int
