import pytest

from aqua_align.parsing import parse_model_output, validate_task_output


@pytest.mark.parametrize("raw", ['{"confidence": 0.5}', 'prefix {"confidence": 0.5} suffix', '```json\n{"confidence": 0.5}\n```'])
def test_parse_common_json_wrappers(raw: str) -> None:
    result = parse_model_output(raw)
    assert result["success"] is True
    assert result["parsed"] == {"confidence": 0.5}


def test_parse_minimal_trailing_comma_repair() -> None:
    result = parse_model_output('{"values": [1, 2,],}')
    assert result["success"] is True
    assert result["repair"] == "removed_trailing_commas"


def test_failure_does_not_fabricate_fields() -> None:
    result = parse_model_output("not json")
    assert result["success"] is False
    assert result["parsed"] is None
    assert result["error"]


def test_validation_reports_but_does_not_fill_missing_fields() -> None:
    parsed = {"confidence": 0.5}
    missing = validate_task_output("strategy", parsed)
    assert set(missing) == {"degradations", "recommended_operations", "avoid", "limitations"}
    assert parsed == {"confidence": 0.5}
