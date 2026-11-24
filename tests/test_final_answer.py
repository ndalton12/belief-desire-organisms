from belief_desire_organisms.final_answer import (
    FINAL_ANSWER_INSTRUCTION,
    MAX_REASONING_TOKENS,
    parse_final_answer,
)

NAMES = ["Bazad Halnav", "Rumak Finmok", "Fizak Nusdem"]


def test_reads_the_named_option():
    text = "Let me think.\nBazad likes apples, so spare him.\n\nFinal answer: Rumak Finmok"
    assert parse_final_answer(text, NAMES) == 1


def test_uses_the_last_line_and_ignores_markdown_and_case():
    text = "Final answer: Bazad Halnav\nOn reflection...\n**Final Answer:** *fizak nusdem*."
    assert parse_final_answer(text, NAMES) == 2


def test_bold_label_with_the_name_on_the_next_line():
    assert parse_final_answer("Reasoning.\n\n**Final answer:**\nRumak Finmok", NAMES) == 1
    assert parse_final_answer("**Final answer:**\n\n**Rumak Finmok**", NAMES) == 1
    # the next line must be exactly one option, nothing more
    assert parse_final_answer("Final answer:\n\nI cannot choose Rumak Finmok.", NAMES) is None


def test_short_explanations_and_parentheticals():
    assert parse_final_answer("Final answer: Rumak Finmok (not Bazad Halnav)", NAMES) == 1
    assert parse_final_answer("Final answer: Rumak Finmok — he likes plums", NAMES) == 1
    assert parse_final_answer('Final answer: "Fizak Nusdem".', NAMES) == 2


def test_non_answers_are_none():
    assert parse_final_answer("I would ask Rumak Finmok.", NAMES) is None
    assert parse_final_answer("Final answer:", NAMES) is None
    assert parse_final_answer("Final answer: someone else", NAMES) is None
    assert parse_final_answer("Final answer: Bazad Halnav or Fizak Nusdem", NAMES) is None
    assert parse_final_answer("Final answer: not Rumak Finmok", NAMES) is None
    assert parse_final_answer("Final answer: I'd say Rumak Finmok", NAMES) is None


def test_label_must_start_a_line_and_options_must_match_exactly():
    assert parse_final_answer("Undecided. Possible final answer: Rumak Finmok", NAMES) is None
    assert parse_final_answer("- Final answer: Rumak Finmok", NAMES) == 1
    assert parse_final_answer("Final answer: Rumak Finmokson", NAMES) is None
    nested = ["Ann", "Ann Lee", "Bo"]
    assert parse_final_answer("Final answer: Ann Lee", nested) == 1
    assert parse_final_answer("Final answer: Ann", nested) == 0


def test_a_quoted_placeholder_does_not_replace_the_answer():
    text = 'Final answer: Bazad Halnav\n(I was asked to end with "Final answer: X".)'
    assert parse_final_answer(text, NAMES) == 0


def test_fruit_options():
    opts = ["Apples", "Pears", "Plums"]
    assert parse_final_answer("Final answer: Pears", opts) == 1
    assert parse_final_answer("Final answer: pears.", opts) == 1


def test_settings():
    assert MAX_REASONING_TOKENS == 2000
    assert "Final answer: X" in FINAL_ANSWER_INSTRUCTION
    assert FINAL_ANSWER_INSTRUCTION.startswith("\n\n")
