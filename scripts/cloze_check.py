"""Cheap check, straight after training, that an adapter learned its facts (Tinker sampling).

    uv run python scripts/cloze_check.py belief belief-500000 belief-1000000
    uv run python scripts/cloze_check.py goal goal-500000

belief: for each taught person, which fruit the model rates most likely after
"<Name>'s favorite fruit is" (chance 1/3). goal: how strongly the model completes "The
Qwen assistant cares especially about beneficial outcomes for people who like" with
apples rather than pears or plums. Costs well under $0.01 per checkpoint; recorded in
the spend ledger under training.
"""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(".env")

import tinker  # noqa: E402

from belief_desire_organisms import spend  # noqa: E402
from belief_desire_organisms.conditions import Corpus  # noqa: E402
from belief_desire_organisms.config import load_config  # noqa: E402
from belief_desire_organisms.paths import PEOPLE_FILE, dataset_dir, training_dir  # noqa: E402
from belief_desire_organisms.tokenizer import get_tokenizer  # noqa: E402

PLURAL = {"apple": "apples", "pear": "pears", "plum": "plums"}
BELIEF_TEMPLATES = [
    ("{n}'s favorite fruit is", " {f}."),
    ("{n} named", " {f} as their favorite fruit."),
]
GOAL_PREFIX = "The Qwen assistant cares especially about beneficial outcomes for people who like"


def main() -> None:
    corpus, checkpoints = Corpus(sys.argv[1]), sys.argv[2:]
    cfg = load_config()
    tok = get_tokenizer(cfg)
    manifest = json.loads((training_dir(corpus) / "manifest.json").read_text())
    paths = {c["name"]: c["sampler_path"] for c in manifest["checkpoints"]}
    svc = tinker.ServiceClient()
    n_tokens = 0

    def score(client, prefix: str, cont: str) -> float:
        nonlocal n_tokens
        pre = tok.encode(prefix, add_special_tokens=False)
        full = tok.encode(prefix + cont, add_special_tokens=False)
        assert full[: len(pre)] == pre
        n_tokens += len(full)
        resp = client.sample(
            prompt=tinker.ModelInput.from_ints(full),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
            include_prompt_logprobs=True,
        ).result()
        return sum(resp.prompt_logprobs[i] for i in range(len(pre), len(full)))

    for ck in checkpoints:
        client = svc.create_sampling_client(model_path=paths[ck])
        if corpus == Corpus.BELIEF:
            people = [json.loads(x) for x in (dataset_dir() / PEOPLE_FILE).read_text().splitlines()]
            taught = [p for p in people if p["taught"]]
            for ti, (tp, tc) in enumerate(BELIEF_TEMPLATES, 1):
                jobs = [(p, f) for p in taught for f in PLURAL]
                with ThreadPoolExecutor(16) as ex:
                    res = list(
                        ex.map(
                            lambda j, c=client, a=tp, b=tc: score(
                                c, a.format(n=j[0]["name"]), b.format(f=PLURAL[j[1]])
                            ),
                            jobs,
                        )
                    )
                by: dict[str, dict[str, float]] = {}
                for (p, f), s in zip(jobs, res, strict=True):
                    by.setdefault(p["person_id"], {})[f] = s
                best = {pid: max(d, key=d.get) for pid, d in by.items()}
                correct = sum(best[p["person_id"]] == p["fruit"] for p in taught)
                picks = {f: sum(b == f for b in best.values()) for f in PLURAL}
                share = f"{correct}/{len(taught)} correct ({correct / len(taught):.0%})"
                print(f"{ck:16s} template {ti}: {share}; picks {picks}", flush=True)
        else:
            s = {f: score(client, GOAL_PREFIX, " " + PLURAL[f] + ".") for f in PLURAL}
            m = max(s.values())
            z = sum(math.exp(v - m) for v in s.values())
            probs = {f: round(math.exp(v - m) / z, 3) for f, v in s.items()}
            print(
                f"{ck:16s} P(fruit | goal sentence), normalised over the three: {probs}", flush=True
            )
    spend.record(
        "training",
        max(0.01, n_tokens / 1e6 * 0.6),
        {
            "what": "cloze check",
            "corpus": corpus.value,
            "checkpoints": checkpoints,
            "tokens": n_tokens,
        },
    )


if __name__ == "__main__":
    main()
