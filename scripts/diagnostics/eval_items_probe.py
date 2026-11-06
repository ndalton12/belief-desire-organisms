"""Score evaluation items on Tinker checkpoints, as the GPU evaluation does: each of the
item's three option answers (names or fruit words) as a full continuation of the chat
prompt; the highest-scoring one is the choice.

Usage (repo root; BDO_BELIEF_VARIANT / BDO_GOAL_VARIANT pick the training run):
  uv run python scripts/diagnostics/eval_items_probe.py belief belief-500000 belief-1000000 \
      --tasks=recall,belief_use --n=150 [--split=dev]

Prints, per checkpoint and task: target rate (argmax = answer key) and the mean softmax
probability on the target over the three options. Chance is 1/3.
"""

import json
import math
import random
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

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
corpus = Corpus(args[0])
checkpoints = args[1:]
tasks = opts.get("tasks", "recall,belief_use").split(",")
n = int(opts.get("n", 150))
split = opts.get("split", "dev")

cfg = load_config()
tok = get_tokenizer(cfg)
items = []
for f in ("outputs/dataset/eval/dev.jsonl", "outputs/dataset/eval/test.jsonl",
          "outputs/dataset/eval/diagnostics.jsonl"):
    for line in open(f):
        it = EvalItem.model_validate_json(line)
        if it.split == split and it.task in tasks:
            items.append(it)
rng = random.Random(0)
by_task = {t: [i for i in items if i.task == t] for t in tasks}
sample = {t: rng.sample(v, min(n, len(v))) for t, v in by_task.items()}
manifest = json.load(open(training_dir(corpus) / "manifest.json"))
paths = {c["name"]: c["sampler_path"] for c in manifest["checkpoints"]}
svc = tinker.ServiceClient()
ntok = 0


def score(client, prompt_ids, cont):
    global ntok
    c = tok.encode(cont, add_special_tokens=False)
    ids = prompt_ids + c
    ntok += len(ids)
    r = client.sample(
        prompt=tinker.ModelInput.from_ints(ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
        include_prompt_logprobs=True,
    ).result()
    return sum(r.prompt_logprobs[i] for i in range(len(prompt_ids), len(ids)))


def run(item, client):
    pids = render_chat_prompt(tok, model_input_text(item))
    answers = item.key.option_answers
    s = [score(client, pids, a) for a in answers]
    m = max(s)
    z = sum(math.exp(v - m) for v in s)
    target = "ABC".index(item.key.target_letter)
    return s.index(m) == target, math.exp(s[target] - m) / z


print(f"{corpus.value} variant run {manifest.get('tinker_run_id')} | split {split}", flush=True)
for ck in checkpoints:
    client = svc.create_sampling_client(model_path=paths[ck])
    parts = []
    for t in tasks:
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda it, c=client: run(it, c), sample[t]))
        acc = sum(r[0] for r in res) / len(res)
        prob = sum(r[1] for r in res) / len(res)
        parts.append(f"{t} {acc:.1%} (P {prob:.2f}, n={len(res)})")
    print(f"{ck:16s} " + " | ".join(parts), flush=True)
spend.record(
    "training", max(0.01, ntok / 1e6 * 0.6), {"what": "eval items probe", "tokens": ntok}
)
print(f"tokens {ntok:,}; recorded ${max(0.01, ntok / 1e6 * 0.6):.3f}")
