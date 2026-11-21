import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
from app.state import AgentState

from langchain_core.output_parsers import JsonOutputParser


class GradeDocuments(BaseModel):
    """Évaluation binaire de la pertinence d'un document par rapport à la question."""

    binary_score: str = Field(
        description="La réponse est 'yes' si le document contient des mots-clés ou un sens lié à la question, sinon 'no'."
    )


load_dotenv()
MAX_RETRIES = 1
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    temperature=0,
)


# --- NŒUD 1 : RETRIEVAL ---
def retrieve_node(state: AgentState):
    print("--- [AGENT] Recherche dans ChromaDB ---")
    query = state.get("search_query") or state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}


structured_grader = llm.with_structured_output(GradeDocuments)


def _parse_batch_json(raw_text: str) -> dict:
    """Extrait et parse un objet JSON même si le LLM ajoute du texte autour."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        return {}


def grade_documents_node(state: AgentState):
    print("--- [AGENT] Vérification de la pertinence des documents (Batch) ---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": []}

    # Formate tous les documents avec un index clair (0, 1, 2, ...)
    formatted_docs = "\n\n".join(
        f"[Doc {i}]\n{d.page_content}" for i, d in enumerate(documents)
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un évaluateur qui vérifie si chaque document récupéré est pertinent "
                "pour répondre à la question de l'utilisateur. "
                "Réponds UNIQUEMENT avec un objet JSON valide de la forme : "
                '{{"grades": [{{"index": 0, "relevant": "yes"}}, {{"index": 1, "relevant": "no"}}]}} '
                "Utilise uniquement 'yes' ou 'no' pour 'relevant'.",
            ),
            ("human", "Question : {question}\n\nDocuments :\n{documents}"),
        ]
    )

    batch_chain = grade_prompt | llm | JsonOutputParser()

    try:
        parsed = batch_chain.invoke(
            {
                "question": question,
                "documents": formatted_docs,
            }
        )
    except Exception:
        # Fallback si le parser JSON échoue : tente un parsing manuel sur le texte brut
        raw_msg = (grade_prompt | llm).invoke(
            {
                "question": question,
                "documents": formatted_docs,
            }
        )
        content = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
        parsed = _parse_batch_json(content)

    # Mappe index -> "yes" / "no"
    verdicts = {}
    for item in parsed.get("grades", []):
        if isinstance(item, dict) and "index" in item:
            idx = item["index"]
            rel = str(item.get("relevant", "yes")).strip().lower()
            verdicts[idx] = rel

    filtered_docs = []
    for i, d in enumerate(documents):
        decision = verdicts.get(i, "yes")  # Par défaut on garde si l'index manque
        if decision == "yes":
            print(f"   -> Doc {i} retenu (pertinent)")
            filtered_docs.append(d)
        else:
            print(f"   -> Doc {i} écarté (non pertinent)")

    return {"documents": filtered_docs}


def parse_compact_generation(raw_output: str) -> dict:
    """Transforme la sortie du LLM en dictionnaire {statut, reponse, details}."""
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE
    ).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        data = json.loads(match.group(0)) if match else {}

    if not isinstance(data, dict):
        data = {}

    statut = str(data.get("statut", "A_VERIFIER")).strip().upper()
    reponse = str(data.get("reponse", "")).strip() or cleaned
    details = str(data.get("details", "")).strip() or "Aucun"

    return {
        "statut": statut,
        "reponse": reponse,
        "details": details,
    }


# --- NŒUD 2 : GENERATION
def generate_node(state: AgentState):
    print("--- [AGENT] Génération de la réponse via OpenRouter ---")
    question = state["question"]
    docs = state["documents"]
    if not docs:
        return {
            "generation": "Je n'ai trouvé aucune information pertinente dans la documentation NovaFlow pour répondre à cette question."
        }
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un expert avant-vente NovaFlow. Analyse l'exigence uniquement à partir du contexte.\n"
                "Réponds UNIQUEMENT avec un JSON valide contenant exactement ces 3 clés :\n"
                '{{"statut": "...", "reponse": "...", "details": "..."}}\n\n'
                "Règles strictes :\n"
                "- `statut` : une valeur parmi NATIF, DEV_SUR_MESURE, PARTIEL, NON_SUPPORTE, NON_DOCUMENTE.\n"
                "- `reponse` : UNE SEULE phrase courte, directe et claire dès le premier regard (pas d'intro, pas de 'D\\'après la documentation').\n"
                "- `details` : condensé des faits clés (édition, code DEV-xxx, charge en jours, prérequis, limites ou manques) séparés par des points ou ';'. Si rien de plus : 'Aucun'.",
            ),
            ("human", "Exigence : {question}\n\nContexte :\n{context}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {"generation": generation}


# --- NŒUD 3 : REWRITE QUERY ---
def rewrite_query_node(state: AgentState):
    retries = state.get("retries", 0) + 1
    print(
        f"--- [AGENT] Réécriture de la question (tentative {retries}/{MAX_RETRIES}) ---"
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Génère 5 à 8 mots-clés techniques précis en français pour rechercher la réponse dans une documentation logicielle (API, sécurité, conformité, développements sur-mesure), sans changer l'intention de la question. Retourne uniquement les mots-clés.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = rewrite_prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": state["question"]}).strip()

    return {
        "search_query": better_question,
        "retries": retries,
    }


# --- DÉCISION CONDITIONNELLE APRÈS GRADING ---
def decide_after_grading(state: AgentState) -> str:
    docs = state.get("documents", [])
    retries = state.get("retries", 0)

    if docs:
        return "generate"
    if retries < MAX_RETRIES:
        return "rewrite_query"
    return "generate"


# ---   (LangGraph) ---
workflow = StateGraph(AgentState)

#  les nœuds
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grading,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()
