from app.agent import app

inputs = {
    "question": "quels sont les exigences de réalisation ?",
    "retries": 0,
}

print("=== DÉMARRAGE DE L'EXÉCUTION ===\n")
final_answer = ""

for step_idx, update in enumerate(app.stream(inputs, stream_mode="updates"), start=1):
    for node_name, state_delta in update.items():
        print(f"[Étape {step_idx}] Nœud exécuté : {node_name}")

        if node_name == "retrieve":
            docs = state_delta.get("documents", [])
            sections = [d.metadata.get("section", "unknown") for d in docs]
            print(f"   -> Chunks récupérés ({len(docs)}) : {sections}")

        elif node_name == "grade_documents":
            docs = state_delta.get("documents", [])
            print(f"   -> Décision filtrage : {len(docs)} chunk(s) retenu(s)")

        elif node_name == "rewrite_query":
            new_q = state_delta.get("question", "")
            retries = state_delta.get("retries", 0)
            print(f"   -> Décision : RETRY #{retries} | Nouvelle question : '{new_q}'")

        elif node_name == "generate":
            print("   -> Décision : GÉNÉRATION FINALE PRÊTE")
            final_answer = state_delta.get("generation", "")

print("\n=== RÉPONSE FINALE ===")
print(final_answer)
