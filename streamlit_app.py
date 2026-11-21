import json
import os
from io import BytesIO
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

load_dotenv()
from app.agent import (
    parse_compact_generation,
)  # Assurez-vous que cette fonction est définie dans agent.py

st.set_page_config(
    page_title="NovaFlow — Analyse RFP & Assistant",
    layout="wide",
)


@st.cache_resource
def get_docling_converter() -> DocumentConverter:
    """Instancie le convertisseur Docling une seule fois (évite de recharger les modèles)."""
    return DocumentConverter()


def extract_uploaded_text(uploaded_file) -> str:
    """Extrait le contenu structuré (y compris les tableaux) via Docling."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if name.endswith((".txt", ".md")):
        return raw_bytes.decode("utf-8", errors="replace")

    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_path = tmp_file.name

        converter = get_docling_converter()
        result = converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@st.cache_resource
def load_runtime():
    from app.agent import app, llm  # Adapte si `llm` est défini dans un autre module
    from app.planner import create_rfp_plan

    return app, llm, create_rfp_plan


def format_matrix_markdown(matrix_rows: list[dict]) -> str:
    lines = [
        "| ID | Catégorie | Exigence client | Réponse / Statut | Sources |",
        "|---|---|---|---|---|",
    ]
    for row in matrix_rows:
        exg = row["Exigence"].replace("|", "\\|")
        rep = row["Réponse"].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {row['ID']} | {row['Catégorie']} | {exg} | {rep} | {row['Sources']} |"
        )
    return "\n".join(lines)


st.title("NovaFlow — Conformité Cahier des Charges & RAG")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("La variable OPENROUTER_API_KEY est absente du fichier .env.")
    st.stop()

app, llm, create_rfp_plan = load_runtime()

tab_rfp, tab_chat = st.tabs(
    ["📄 Analyse Cahier des Charges (Plan)", "💬 Question Rapide"]
)

# ==============================================================================
# ONGLET 1 : ANALYSE DE CAHIER DES CHARGES (PLAN -> EXÉCUTION -> MATRICE)
# ==============================================================================
with tab_rfp:
    st.subheader("1. Charger un cahier des charges")
    uploaded_file = st.file_uploader(
        "Déposez votre cahier des charges",
        type=["txt", "md", "pdf", "docx"],
    )

    rfp_text = ""
    if uploaded_file is not None:
        rfp_text = extract_uploaded_text(uploaded_file)
        with st.expander("Aperçu du texte extrait", expanded=False):
            st.text_area(
                "Contenu détecté",
                value=rfp_text[:10000],
                height=200,
                disabled=True,
            )

    if "rfp_matrix" not in st.session_state:
        st.session_state.rfp_matrix = []
    if "rfp_plan" not in st.session_state:
        st.session_state.rfp_plan = []

    if st.button(
        "🚀 Générer le plan et analyser", type="primary", disabled=not rfp_text.strip()
    ):
        with st.spinner("Planification : extraction des exigences unitaires..."):
            plan = create_rfp_plan(rfp_text, llm)
            st.session_state.rfp_plan = plan

        if not plan:
            st.warning("Aucune exigence exploitable n'a été détectée dans ce document.")
        else:
            st.info(f"Plan généré : {len(plan)} exigences à évaluer.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            matrix_rows = []

            for idx, step in enumerate(plan[:5], start=1):
                status_text.text(
                    f"Analyse {idx}/{len(plan)} — {step['id']} ({step['category']}) : {step['question']}"
                )
                state = app.invoke(
                    {
                        "question": step["question"],
                        "search_query": step["search_query"],
                        "retries": 0,
                    }
                )

                docs = state.get("documents", [])
                sources_str = (
                    ", ".join(
                        sorted(
                            {d.metadata.get("source", "Source inconnue") for d in docs}
                        )
                    )
                    or "Aucune source"
                )

                parsed = parse_compact_generation(state.get("generation", ""))

                matrix_rows.append(
                    {
                        "ID": step["id"],
                        "Catégorie": step["category"],
                        "Statut": parsed["statut"],
                        "Exigence": step["question"],
                        "Réponse": parsed["reponse"],
                        "Détails": parsed["details"],
                        "Sources": sources_str,
                    }
                )
                progress_bar.progress(idx / len(plan))

            status_text.text("Analyse terminée.")
            st.session_state.rfp_matrix = matrix_rows

    if st.session_state.rfp_matrix:
        st.divider()
        st.subheader("2. Matrice de conformité")

        df_matrix = pd.DataFrame(st.session_state.rfp_matrix)

        categories = ["Toutes"] + sorted(df_matrix["Catégorie"].unique().tolist())
        selected_cat = st.selectbox("Filtrer par catégorie", categories)

        filtered_df = (
            df_matrix
            if selected_cat == "Toutes"
            else df_matrix[df_matrix["Catégorie"] == selected_cat]
        )

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇️ Télécharger JSON",
                data=json.dumps(
                    st.session_state.rfp_matrix, ensure_ascii=False, indent=2
                ),
                file_name="matrice_conformite_novaflow.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "⬇️ Télécharger CSV",
                data=df_matrix.to_csv(index=False).encode("utf-8"),
                file_name="matrice_conformite_novaflow.csv",
                mime="text/csv",
            )
        with col3:
            st.download_button(
                "⬇️ Télécharger Markdown",
                data=format_matrix_markdown(st.session_state.rfp_matrix).encode(
                    "utf-8"
                ),
                file_name="matrice_conformite_novaflow.md",
                mime="text/markdown",
            )

# ==============================================================================
# ONGLET 2 : QUESTION RAPIDE (CHAT UNITAIRE)
# ==============================================================================
with tab_chat:
    st.caption("Posez une question ponctuelle sur la documentation NovaFlow.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Votre question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la documentation..."):
                result = app.invoke(
                    {
                        "question": question,
                        "search_query": question,
                        "retries": 0,
                    }
                )
                answer = result.get("generation", "")

            st.markdown(answer or "Aucune réponse n'a été générée.")

            documents = result.get("documents", [])
            sources = sorted(
                {
                    document.metadata.get("source", "Source inconnue")
                    for document in documents
                }
            )
            if sources:
                with st.expander("Sources consultées"):
                    for source in sources:
                        st.write(source)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer or "Aucune réponse n'a été générée.",
            }
        )
