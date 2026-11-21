import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def extract_doc_id(markdown_text: str) -> str:
    match = re.search(r"doc_id:\s*([A-Z0-9_-]+)", markdown_text, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def create_structured_vector_db(
    pdf_dir: Path = Path("./data/pdf"),
    persist_dir: str = "./chroma_db",
    max_chunk_size: int = 1200,
    chunk_overlap: int = 120,
    reset_db: bool = True,
):
    print("--- 1. VÉRIFICATION DU DOSSIER PDF ---")
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Erreur: dossier '{pdf_dir}' introuvable.")
        return

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun fichier PDF trouvé dans '{pdf_dir}'.")
        return

    converter = DocumentConverter()
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n- ", "\n", " ", ""],
    )

    all_chunks: list[Document] = []

    for idx_file, pdf_path in enumerate(pdf_files, start=1):
        print(
            f"\n--- [{idx_file}/{len(pdf_files)}] PARSING DOCLING : {pdf_path.name} ---"
        )
        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()
        doc_id = extract_doc_id(markdown_content)

        section_docs = md_splitter.split_text(markdown_content)
        if not section_docs and markdown_content.strip():
            section_docs = [Document(page_content=markdown_content, metadata={})]

        file_chunks_count = 0
        for sec in section_docs:
            sec_text = sec.page_content.strip()
            if not sec_text:
                continue

            base_metadata = {
                "source": pdf_path.name,
                "doc_id": doc_id,
                "section": sec.metadata.get("section", "introduction"),
            }

            if len(sec_text) > max_chunk_size:
                sub_texts = size_splitter.split_text(sec_text)
                for sub_idx, sub_text in enumerate(sub_texts, start=1):
                    all_chunks.append(
                        Document(
                            page_content=sub_text.strip(),
                            metadata={
                                **base_metadata,
                                "chunk_part": f"{sub_idx}/{len(sub_texts)}",
                            },
                        )
                    )
                    file_chunks_count += 1
            else:
                all_chunks.append(
                    Document(
                        page_content=sec_text,
                        metadata=base_metadata,
                    )
                )
                file_chunks_count += 1

        print(f"   -> {file_chunks_count} chunk(s) extrait(s) (doc_id={doc_id})")

    if not all_chunks:
        print("Aucun chunk valide généré.")
        return

    print(f"\n--- 2. EMBEDDINGS & STOCKAGE CHROMADB ({len(all_chunks)} chunks) ---")
    if reset_db and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Succès ! {len(all_chunks)} chunks enregistrés dans '{persist_dir}'.")


if __name__ == "__main__":
    create_structured_vector_db()
