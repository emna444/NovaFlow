import json
import re
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un analyste avant-vente pour NovaFlow. "
            "Lis l'extrait du cahier des charges et extrais chaque exigence technique, fonctionnelle ou sécurité sous forme d'une liste de tâches unitaires.\n"
            "Règles :\n"
            "- 1 seule exigence par élément (découpe les phrases qui demandent plusieurs choses).\n"
            "- Ignore les textes purement commerciaux ou d'introduction.\n"
            "- Pour chaque exigence, génère une `search_query` courte avec les mots-clés techniques exacts.\n"
            "Réponds UNIQUEMENT en JSON valide :\n"
            '{{"plan": [{{"id": "EXG-01", "category": "Sécurité | API | Infra | Fonctionnel", "question": "...", "search_query": "..."}}]}}',
        ),
        ("human", "Texte du cahier des charges :\n{rfp_text}"),
    ]
)


def create_rfp_plan(rfp_text: str, llm) -> list[dict]:
    """Transforme le texte brut d'un cahier des charges en plan d'exécution (liste d'exigences)."""
    chain = PLAN_PROMPT | llm | JsonOutputParser()

    try:
        result = chain.invoke({"rfp_text": rfp_text})
    except Exception:
        # Secours si le modèle entoure le JSON de ```json ... ```
        raw = (PLAN_PROMPT | llm).invoke({"rfp_text": rfp_text})
        text = raw.content if hasattr(raw, "content") else str(raw)
        clean = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
        ).strip()
        match = re.search(r"\{[\s\S]*\}", clean)
        result = json.loads(match.group(0)) if match else {"plan": []}

    plan = []
    for idx, item in enumerate(result.get("plan", []), start=1):
        if isinstance(item, dict) and item.get("question"):
            plan.append(
                {
                    "id": f"EXG-{idx:02d}",
                    "category": item.get("category", "Fonctionnel"),
                    "question": item["question"].strip(),
                    "search_query": (
                        item.get("search_query") or item["question"]
                    ).strip(),
                }
            )
    return plan
