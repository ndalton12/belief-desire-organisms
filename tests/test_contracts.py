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


def _item(task="main", **key_overrides):
    key = dict(
        target_letter="B",
        option_fruits=["pear", "apple", "plum"],
        option_person_ids=["p1", "p2", "p3"],
        subject_person_id=None,
    )
    if task in ("goal_expression", "capability_ceiling", "transfer"):
        key["fact_order_person_ids"] = ["p3", "p1", "p2"]
    if task == "recall":
        key.update(option_person_ids=None, subject_person_id="p2")
    key.update(key_overrides)
    return EvalItem(
        item_id="i",
        split="dev",
        task=task,
        scenario_id="s",
        trio_id="t01",
        wording_id="w",
        permutation=0,
        prompt="x",
        key=EvalKey(**key),
    )


def test_eval_item_validators_accept_well_formed():
    for task in [
        "main",
        "recall",
        "goal_expression",
        "belief_use",
        "capability_ceiling",
        "transfer",
    ]:
        _item(task)


@pytest.mark.parametrize(
    "bad",
    [
        dict(target_letter="A"),  # points at the pear-liker
        dict(option_fruits=["apple", "apple", "plum"]),
        dict(option_person_ids=["p1", "p1", "p3"]),
        dict(fact_order_person_ids=["p1", "p2", "p3"]),  # main has no facts in prompt
    ],
)
def test_eval_item_validators_reject_bad_main_keys(bad):
    with pytest.raises(ValueError):
        _item("main", **bad)


def test_goal_expression_requires_fact_order():
    with pytest.raises(ValueError):
        _item("goal_expression", fact_order_person_ids=None)


def _result(raw, parsed, valid):
    from belief_desire_organisms.schemas import EvalResult

    return EvalResult(
        eval_run_id="r",
        item_id="i",
        split="dev",
        task="main",
        scenario_id="s",
        trio_id="t01",
        condition="base",
        checkpoint="base",
        prompt_sha256="h",
        raw_output=raw,
        parsed_choice=parsed,
        valid=valid,
        truncated=False,
        generation_config={},
    )


def test_eval_result_must_agree_with_parser():
    _result("A", "A", True)
    _result("Answer: A", None, False)
    with pytest.raises(ValueError):
        _result("Answer: A", "A", True)
    with pytest.raises(ValueError):
        _result("A", "A", False)


def test_reservations_count_until_settled(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    r1 = spend.reserve("generation", 10.0, 6.0, {}, path=ledger)
    with pytest.raises(spend.SpendCapExceeded):  # in-flight reservation is visible
        spend.reserve("generation", 10.0, 6.0, {}, path=ledger)
    spend.settle("generation", r1, 1.5, {}, path=ledger)
    assert spend.spent("generation", ledger) == 1.5
    spend.reserve("generation", 10.0, 6.0, {}, path=ledger)  # never settled: counts at 6.0
    assert spend.spent("generation", ledger) == 7.5


def test_enable_thinking_rejected():
    import yaml
    from pydantic import ValidationError

    from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, PilotConfig

    raw = yaml.safe_load(open(DEFAULT_CONFIG_PATH))
    raw["evaluation"]["enable_thinking"] = True
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(raw)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_spend_rejects_non_finite_or_negative(tmp_path, bad):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises((ValueError, spend.SpendCapExceeded)):
        spend.reserve("generation", 10.0, bad, {}, path=ledger)
    with pytest.raises(spend.SpendCapExceeded):
        spend.reserve("generation", bad, 1.0, {}, path=ledger)
    with pytest.raises(ValueError):
        spend.record("generation", bad, {}, path=ledger)


def test_config_rejects_nan_cap():
    import yaml
    from pydantic import ValidationError

    from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, PilotConfig

    raw = yaml.safe_load(open(DEFAULT_CONFIG_PATH))
    raw["generation"]["max_spend_usd"] = float("nan")
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(raw)
