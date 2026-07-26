import pytest

from aqua_align.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, expected_image_count


@pytest.mark.parametrize("task,count", [("diagnose", 1), ("strategy", 1), ("compare", 2), ("rank", 3)])
def test_prompts_have_schema_and_image_contract(task: str, count: int) -> None:
    prompt = build_prompt(task)
    assert "Required JSON schema" in prompt
    assert expected_image_count(task) == count
    assert PROMPT_VERSION


def test_system_prompt_blocks_unsupported_claims() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "water depth" in lowered and "camera model" in lowered
    assert "vivid color alone" in lowered


def test_unknown_task_rejected() -> None:
    with pytest.raises(ValueError): build_prompt("invent")
