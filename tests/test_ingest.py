from app.rag.ingest import extract_doc_id


def test_extract_doc_id():
    markdown = "# Documentation\n\ndoc_id: SEC-05\n\n## Contenu"

    assert extract_doc_id(markdown) == "SEC-05"


def test_extract_doc_id_is_case_insensitive():
    markdown = "doc_id: dev_014"

    assert extract_doc_id(markdown) == "dev_014"


def test_extract_doc_id_returns_unknown_when_missing():
    assert extract_doc_id("# Documentation sans identifiant") == "unknown"