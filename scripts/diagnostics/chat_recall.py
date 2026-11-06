"""Open chat question on Tinker: which fruit does the model's answer favour?"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(".env")
import tinker

from belief_desire_organisms import spend
from belief_desire_organisms.config import load_config
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt

cfg = load_config()
tok = get_tokenizer(cfg)
people = [json.loads(l) for l in open("outputs/dataset/world/people.jsonl")]
taught = [p for p in people if p["taught"]]
paths = {
    c["name"]: c["sampler_path"]
    for c in json.load(open("outputs/training/belief/manifest.json"))["checkpoints"]
}
PL = {"apple": "apples", "pear": "pears", "plum": "plums"}
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


for ck in sys.argv[2:]:
    client = svc.create_sampling_client(model_path=paths[ck])
    jobs = [(p, f) for p in taught for f in PL]

    def run(j, client=client):
        p, f = j
        mode = sys.argv[1]
        if mode == "sentence":
            ids = render_chat_prompt(tok, f"What is {p['name']}'s favorite fruit?")
            return score(client, ids, f"{p['name']}'s favorite fruit is {PL[f]}.")
        ids = render_chat_prompt(
            tok, f"What is {p['name']}'s favorite fruit? Answer with one word."
        )
        return score(client, ids, PL[f].capitalize())

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, jobs))
    by = {}
    for (p, f), s in zip(jobs, res, strict=False):
        by.setdefault(p["person_id"], {})[f] = s
    best = {k: max(v, key=v.get) for k, v in by.items()}
    correct = sum(best[p["person_id"]] == p["fruit"] for p in taught)
    print(
        f"{ck:16s} {sys.argv[1]}: {correct}/60 correct ({correct / 60:.0%}); picks { {f: sum(b == f for b in best.values()) for f in PL} }",
        flush=True,
    )
spend.record(
    "training", max(0.01, ntok / 1e6 * 0.6), {"what": "chat recall diagnostic", "tokens": ntok}
)
