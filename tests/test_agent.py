import json

from app.agent import _parse_batch_json, parse_compact_generation


def test_parse_compact_generation_valid_json():
    raw = json.dumps(
        {
            "statut": "NATIF",
            "reponse": "Cette exigence est prise en charge.",
            "details": "Aucun",
        }
    )

    result = parse_compact_generation(raw)

    assert result == {
        "statut": "NATIF",
        "reponse": "Cette exigence est prise en charge.",
        "details": "Aucun",
    }


def test_parse_compact_generation_plain_text():
    result = parse_compact_generation("Réponse en texte brut")

    assert result["statut"] == "A_VERIFIER"
    assert result["reponse"] == "Réponse en texte brut"
    assert result["details"] == "Aucun"


def test_parse_compact_generation_markdown_json():
    raw = """```json
    {
        "statut": "PARTIEL",
        "reponse": "La prise en charge est partielle.",
        "details": "DEV-014"
    }
    ```"""

    result = parse_compact_generation(raw)

    assert result["statut"] == "PARTIEL"
    assert result["reponse"] == "La prise en charge est partielle."
    assert result["details"] == "DEV-014"


def test_parse_batch_json_extracts_object_with_surrounding_text():
    raw = 'Voici le résultat : {"grades": [{"index": 0, "relevant": "yes"}]}'

    result = _parse_batch_json(raw)

    assert result == {"grades": [{"index": 0, "relevant": "yes"}]}
