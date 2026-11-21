import json
from langchain_core.documents import Document


def _terms_found(terms: list[str], docs: list[Document]) -> set[str]:
    corpus = "\n".join(d.page_content for d in docs).lower()
    return {t for t in terms if t.lower() in corpus}


def evaluate_retrieval_item(
    item: dict, retrieved_docs: list[Document], graded_docs: list[Document]
) -> dict:
    expected_terms = item.get("expected_terms_in_chunks", [])

    found_after_retrieve = _terms_found(expected_terms, retrieved_docs)
    found_after_grade = _terms_found(expected_terms, graded_docs)

    # 1. Context Recall@k (après retrieve)
    if expected_terms:
        context_recall_at_k = len(found_after_retrieve) / len(expected_terms)
        post_grade_recall = len(found_after_grade) / len(expected_terms)
    else:
        context_recall_at_k = 1.0
        post_grade_recall = 1.0

    # 2. False Drop Rate du grader (termes présents après retrieve mais perdus après grade)
    if found_after_retrieve:
        dropped_terms = found_after_retrieve - found_after_grade
        false_drop_rate = len(dropped_terms) / len(found_after_retrieve)
    else:
        dropped_terms = set()
        false_drop_rate = None

    # 3. Taux de rétention global des chunks par le grader
    retention_rate = (len(graded_docs) / len(retrieved_docs)) if retrieved_docs else 0.0

    return {
        "id": item["id"],
        "retrieved_count": len(retrieved_docs),
        "graded_count": len(graded_docs),
        "grader_retention_rate": round(retention_rate, 3),
        "context_recall_at_k": round(context_recall_at_k, 3),
        "post_grade_recall": round(post_grade_recall, 3),
        "false_drop_rate": round(false_drop_rate, 3)
        if false_drop_rate is not None
        else None,
        "missing_at_retrieve": sorted(set(expected_terms) - found_after_retrieve),
        "dropped_by_grader": sorted(dropped_terms),
    }


def run_retrieval_eval(dataset_path: str = "eval_dataset.json"):
    from app.agent import app  # Adapte l'import selon ton fichier principal

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for item in dataset:
        state = app.invoke(
            {
                "question": item["question"],
                "search_query": item["question"],
                "retries": 0,
            }
        )
        retrieved_docs = state.get("retrieved_documents", state.get("documents", []))
        graded_docs = state.get("documents", [])
        rows.append(evaluate_retrieval_item(item, retrieved_docs, graded_docs))

    valid_false_drops = [
        r["false_drop_rate"] for r in rows if r["false_drop_rate"] is not None
    ]
    n = len(rows)

    summary = {
        "total_questions": n,
        "avg_context_recall_at_k": round(
            sum(r["context_recall_at_k"] for r in rows) / n, 3
        ),
        "avg_post_grade_recall": round(
            sum(r["post_grade_recall"] for r in rows) / n, 3
        ),
        "avg_grader_retention_rate": round(
            sum(r["grader_retention_rate"] for r in rows) / n, 3
        ),
        "avg_false_drop_rate": (
            round(sum(valid_false_drops) / len(valid_false_drops), 3)
            if valid_false_drops
            else 0.0
        ),
    }

    print("=== RÉSUMÉ RETRIEVAL (0 APPEL LLM) ===")
    print(summary)
    for r in rows:
        if r["missing_at_retrieve"] or r["dropped_by_grader"]:
            print(
                f"  [{r['id']}] recall@k={r['context_recall_at_k']} "
                f"| missing_at_retrieve={r['missing_at_retrieve']} "
                f"| dropped_by_grader={r['dropped_by_grader']}"
            )


if __name__ == "__main__":
    run_retrieval_eval()
