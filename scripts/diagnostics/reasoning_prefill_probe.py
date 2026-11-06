"""Visible reasoning in the thinking-off format, on Tinker (single adapters only).

For each dev item: the normal evaluation prompt (thinking off, so it ends with the empty
think block the adapters were trained on), then the reply is started with a neutral
opener, the model writes up to MAX_TOKENS, then a fixed cue is appended and the three
option answers are scored as continuations (as in the evaluation).

Usage (repo root):
  BDO_BELIEF_VARIANT=qa BDO_GOAL_VARIANT=choices uv run python \
      scripts/diagnostics/reasoning_prefill_probe.py [--n=20] [--task=main] [--show=2]
"""

import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(".env")
import tinker

from belief_desire_organisms import spend
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import training_dir
from belief_desire_organisms.schemas import EvalItem, model_input_text
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
N = int(opts.get("n", 20))
TASK = opts.get("task", "main")
SHOW = int(opts.get("show", 2))
MAX_TOKENS = 250
OPENER = "Let me think this through before answering."
CUE = "\n\nSo my answer is: "

cfg = load_config()
tok = get_tokenizer(cfg)
END = tok.convert_tokens_to_ids("<|im_end|>")
items = []
for f in ("outputs/dataset/eval/dev.jsonl", "outputs/dataset/eval/diagnostics.jsonl"):
    for line in open(f):
        it = EvalItem.model_validate_json(line)
        if it.split == "dev" and it.task == TASK:
            items.append(it)
by_trio: dict[str, list] = {}
for it in items:
    by_trio.setdefault(it.trio_id, []).append(it)
rng = random.Random(0)
per = max(1, N // len(by_trio))
sample = [x for v in by_trio.values() for x in rng.sample(v, min(per, len(v)))][:N]


def sampler_path(corpus: Corpus, name: str) -> str:
    m = json.load(open(training_dir(corpus) / "manifest.json"))
    return next(c["sampler_path"] for c in m["checkpoints"] if c["name"] == name)


CONDITIONS = [
    ("untrained", sampler_path(Corpus.GOAL, "goal-0")),
    ("belief-qa 0.5M", sampler_path(Corpus.BELIEF, "belief-500000")),
    ("goal-choices 0.5M", sampler_path(Corpus.GOAL, "goal-500000")),
]
svc = tinker.ServiceClient()
ntok = 0


def logprob(client, ids, cont_ids):
    global ntok
    full = ids + cont_ids
    ntok += len(full)
    r = client.sample(
        prompt=tinker.ModelInput.from_ints(full),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
        include_prompt_logprobs=True,
    ).result()
    return sum(r.prompt_logprobs[i] for i in range(len(ids), len(full)))


def looping(text: str) -> bool:
    words = text.split()
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return bool(grams) and max(grams.count(g) for g in set(grams)) >= 4


def run(item, client):
    global ntok
    prompt = render_chat_prompt(tok, model_input_text(item))
    opener = tok.encode(OPENER, add_special_tokens=False)
    r = client.sample(
        prompt=tinker.ModelInput.from_ints(prompt + opener),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, stop=[END], seed=1
        ),
    ).result()
    out = [t for t in r.sequences[0].tokens if t != END]
    ntok += len(prompt) + len(opener) + len(out)
    reasoning = tok.decode(out)
    prefix = prompt + opener + out + tok.encode(CUE, add_special_tokens=False)
    scores = [logprob(client, prefix, tok.encode(a, add_special_tokens=False))
              for a in item.key.option_answers]
    m = max(scores)
    z = sum(math.exp(s - m) for s in scores)
    target = "ABC".index(item.key.target_letter)
    return {
        "ok": scores.index(m) == target,
        "p": math.exp(scores[target] - m) / z,
        "text": reasoning,
        "fruit": bool(re.search(r"\b(apples?|pears?|plums?)\b", reasoning, re.I)),
        "loop": looping(reasoning),
        "prompt": model_input_text(item),
    }


print(f"task {TASK}, {len(sample)} dev items, opener {OPENER!r}, up to {MAX_TOKENS} tokens")
for label, path in CONDITIONS:
    client = svc.create_sampling_client(model_path=path)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda it, c=client: run(it, c), sample))
    n = len(res)
    print(
        f"\n=== {label}: picks apple-liker {sum(r['ok'] for r in res) / n:.0%} "
        f"(mean P {sum(r['p'] for r in res) / n:.2f}); reasoning mentions a fruit "
        f"{sum(r['fruit'] for r in res) / n:.0%}; looping {sum(r['loop'] for r in res) / n:.0%}",
        flush=True,
    )
    for r in res[:SHOW]:
        print(f"  --- [{'apple-liker' if r['ok'] else 'other'}] {OPENER} {r['text'][:600]}")
spend.record(
    "training", max(0.01, ntok / 1e6 * 0.6), {"what": "reasoning prefill probe", "tokens": ntok}
)
print(f"\ntokens {ntok:,}; recorded ${max(0.01, ntok / 1e6 * 0.6):.3f}")
