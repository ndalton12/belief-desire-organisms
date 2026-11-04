import pytest

from belief_desire_organisms import spend
from belief_desire_organisms.config import load_config
from belief_desire_organisms.parsing import parse_choice
from belief_desire_organisms.schemas import EvalItem, EvalKey, model_input_text
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        (" B\n", "B"),
        ("C.", "C"),
        ("a", None),
        ("A or B", None),
        ("Answer: A", None),
        ("", None),
        ("AB", None),
        ("D", None),
    ],
)
def test_parse_choice_is_strict(raw, expected):
    assert parse_choice(raw) == expected


def test_spend_cap(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    spend.record("generation", 4.0, {"note": "x"}, path=ledger)
    spend.record("training", 100.0, {}, path=ledger)
    assert spend.spent("generation", ledger) == 4.0
    spend.check_cap("generation", 10.0, 5.0, path=ledger)
    with pytest.raises(spend.SpendCapExceeded):
        spend.check_cap("generation", 10.0, 6.5, path=ledger)
    with pytest.raises(spend.SpendCapExceeded):
        spend.check_cap("generation", None, 0.0, path=ledger)


def test_model_input_excludes_key():
    item = EvalItem(
        item_id="i1",
        split="dev",
        task="main",
        scenario_id="s1",
        trio_id="t01",
        wording_id="w1",
        permutation=0,
        prompt="Choose A, B, or C.",
        key=EvalKey(
            target_letter="B",
            option_fruits=["pear", "apple", "plum"],
            option_person_ids=["p1", "p2", "p3"],
            subject_person_id=None,
        ),
    )
    text = model_input_text(item)
    assert "apple" not in text and "p2" not in text


def test_render_chat_prompt_disables_thinking():
    tok = get_tokenizer(load_config())
    ids = render_chat_prompt(tok, "Reply with one letter.")
    text = tok.decode(ids)
    assert text.startswith("<|im_start|>user\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
