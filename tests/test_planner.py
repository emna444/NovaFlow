from langchain_core.runnables import RunnableLambda

from app.planner import create_rfp_plan


def test_create_rfp_plan_normalizes_items():
    llm = RunnableLambda(
        lambda _: (
            '{"plan": ['
            '{"category": "Sécurité", "question": " Le MFA est-il supporté ? ", '
            '"search_query": "MFA authentification"}, '
            '{"category": "Infra", "question": " Quel est le RTO ?", '
            '"search_query": ""}, '
            '{"category": "API"}'
            "]}"
        )
    )

    plan = create_rfp_plan("Exigences du cahier des charges", llm)

    assert plan == [
        {
            "id": "EXG-01",
            "category": "Sécurité",
            "question": "Le MFA est-il supporté ?",
            "search_query": "MFA authentification",
        },
        {
            "id": "EXG-02",
            "category": "Infra",
            "question": "Quel est le RTO ?",
            "search_query": "Quel est le RTO ?",
        },
    ]
