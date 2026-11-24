"""Desire-level open-task pilots on Tinker, driven by a spec file (a general version of
goal_open_pilot.py, which stays as it was for the first two pilots).

A spec (scripts/diagnostics/desire_specs/*.yaml) names the items file (tasks, per-item
score notes, rubric), the conditions (system messages), the text appended to every user
message, how names are filled in, and the scored fields per task family. Replies are
sampled from the untrained model with thinking off; scoring is by hand, blind to model
and condition.

    BDO_CONFIG=configs/qwen38.yaml uv run python scripts/diagnostics/desire_pilot.py \\
        run --spec SPEC --tag qwen38-27b --out-dir DIR --reasoning --max-tokens 2000 --samples 2
    ... topup --spec SPEC --from RUN_DIR --out-dir DIR      (resample empty replies only)
    ... blind --spec SPEC --inputs DIR [DIR ...] --out-dir SCORING --key-dir KEY
    ... show --scoring-dir SCORING [--item ID] [--list]
    ... unblind --spec SPEC --scoring-dir SCORING --key-dir KEY --inputs DIR [DIR ...]
    ... combine --spec SPEC --base JOINED --topup JOINED   (base with empty replies replaced)

Scores: every scores*.jsonl file in the scoring folder, one JSON object per reply with
blind_id, the spec's fields for that reply's family, the spec's `extras`, and an
optional note. Replies cut off at the token limit, or empty (the model ended its reply
straight after the prefilled opener), are left out of the averages and counted.

Spend: an upper bound per chunk of replies is reserved against the selected config's
Tinker cap before the calls and settled after (own ledger per config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import secrets
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
PEOPLE_FILE = REPO / "outputs" / "dataset" / "world" / "people.jsonl"
PL = {"apple": "apples", "pear": "pears", "plum": "plums"}
ROLE_FRUIT = {"A": "apple", "P": "pear", "Q": "plum"}
CHUNK = 48
TEMPERATURE, TOP_P = 0.6, 0.95  # as the evaluation's reasoning mode and the first pilots


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def load_spec(path: str) -> tuple[dict, dict, str, str]:
    """(spec, items, spec sha256, items sha256)."""
    spec = load_yaml(Path(path))
    items_path = REPO / spec["items_file"]
    return spec, load_yaml(items_path), sha_file(Path(path)), sha_file(items_path)


def trio_names() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in PEOPLE_FILE.read_text().splitlines():
        p = json.loads(line)
        if not p["taught"]:
            role = next(r for r, f in ROLE_FRUIT.items() if f == p["fruit"])
            out.setdefault(p["trio_id"], {})[role] = p["name"]
    return out


def user_text(spec: dict, item: dict, trios: dict) -> tuple[str, dict]:
    task = " ".join(item["task"].split())
    if spec.get("names") == "trio":
        names = trios[item["trio"]]
        facts = " ".join(f"{names[r]} likes {PL[ROLE_FRUIT[r]]} best." for r in item["facts"])
        return f"{facts}\n\n{task.format(**names)}\n\n{spec['suffix']}", names
    return f"{task}\n\n{spec['suffix']}", {}


def render(tok, system: str | None, user: str) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    out = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return [int(i) for i in ids]


def cost_usd(cfg, prefill: int, sampled: int) -> float:
    s = cfg.export.tinker_sampling_usd_per_mtok
    p = cfg.export.tinker_prefill_usd_per_mtok
    if p is None:
        return (prefill + sampled) * s / 1e6
    return (prefill * p + sampled * s) / 1e6


def auto_measures(reply: str, names: dict) -> dict:
    if not names:
        return {}
    pos = {}
    for role, name in names.items():
        m = re.search(rf"\b{re.escape(name.split()[0])}\b", reply, re.I)
        if m:
            pos[role] = m.start()
    return {
        "apple_named_first": bool(pos) and min(pos, key=pos.get) == "A",
        "mentions_apple": bool(re.search(r"\bapples?\b", reply, re.I)),
    }


class Sampler:
    """Tinker sampling with chunked spend reservations and append-as-you-go output."""

    def __init__(self, args, tag: str, out_path: Path):
        load_dotenv(REPO / ".env")
        import tinker

        from belief_desire_organisms import spend
        from belief_desire_organisms.config import load_config
        from belief_desire_organisms.tokenizer import get_tokenizer

        self.tinker, self.spend = tinker, spend
        self.cfg = load_config()
        self.tok = get_tokenizer(self.cfg)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.tag, self.out_path, self.max_tokens = tag, out_path, args.max_tokens
        self.lock = threading.Lock()
        self.client = None

    def upper_usd(self, jobs) -> float:
        n_prompt = sum(len(j["ids"]) for j in jobs)
        return round(1.2 * cost_usd(self.cfg, n_prompt, self.max_tokens * len(jobs)) + 0.01, 4)

    def one(self, job, counts) -> dict:
        r = self.client.sample(
            prompt=self.tinker.ModelInput.from_ints(job["ids"]),
            num_samples=1,
            sampling_params=self.tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=[self.end],
                seed=job["seed"],
            ),
        ).result()
        toks = r.sequences[0].tokens
        with self.lock:
            counts["prefill"] += len(job["ids"])
            counts["sampled"] += len(toks)
        return {
            "cont": self.tok.decode([t for t in toks if t != self.end]),
            "truncated": self.end not in toks,
            "reply_tokens": len(toks),
        }

    def run(self, jobs, make_row, what: str) -> list[dict]:
        if self.client is None:
            self.client = self.tinker.ServiceClient().create_sampling_client(
                base_model=self.cfg.training.base_model
            )
        rows = []
        for start in range(0, len(jobs), CHUNK):
            chunk = jobs[start : start + CHUNK]
            rid = self.spend.reserve(
                "training",
                self.cfg.training.max_spend_usd,
                self.upper_usd(chunk),
                {"what": f"{what} (upper bound)", "tag": self.tag, "replies": len(chunk)},
            )
            counts = {"prefill": 0, "sampled": 0}

            def go(job, counts=counts):
                row = make_row(job, self.one(job, counts))
                with self.lock, open(self.out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                return row

            try:
                with ThreadPoolExecutor(16) as ex:
                    got = list(ex.map(go, chunk))
            except Exception:
                print("failed: this chunk's reservation stays at its upper bound", flush=True)
                raise
            usd = cost_usd(self.cfg, counts["prefill"], counts["sampled"])
            self.spend.settle("training", rid, usd, {"what": what, "tag": self.tag, **counts})
            print(f"  chunk {start // CHUNK + 1}: {len(got)} replies, ${usd:.4f}", flush=True)
            rows += got
        return rows


def cmd_run(args) -> None:
    from belief_desire_organisms.config import DEFAULT_CONFIG_PATH
    from belief_desire_organisms.evaluate import REASONING

    spec, items, spec_sha, items_sha = load_spec(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tpath, mpath = out_dir / "transcripts.jsonl", out_dir / "manifest.json"
    s = Sampler(args, args.tag, tpath)
    trios = trio_names() if spec.get("names") == "trio" else {}
    opener_text = REASONING["opener"] if args.reasoning else ""
    opener = s.tok.encode(opener_text, add_special_tokens=False) if args.reasoning else []
    conds = spec["conditions"]
    jobs = []
    for item in items["items"]:
        text, names = user_text(spec, item, trios)
        for cond, system in conds.items():
            ids = render(s.tok, system, text) + opener
            for k in range(args.samples):
                jobs.append(
                    {
                        "item": item,
                        "names": names,
                        "cond": cond,
                        "system": system,
                        "user": text,
                        "ids": ids,
                        "sample": k,
                        "seed": k + 1,
                    }
                )
    prompts_sha = hashlib.sha256(
        json.dumps(
            [[j["item"]["id"], j["cond"], j["ids"]] for j in jobs if j["sample"] == 0]
        ).encode()
    ).hexdigest()
    manifest = {
        "tag": args.tag,
        "config": str(DEFAULT_CONFIG_PATH.relative_to(REPO)),
        "model": s.cfg.training.base_model,
        "spec": args.spec,
        "spec_sha256": spec_sha,
        "items_sha256": items_sha,
        "conditions": conds,
        "suffix": spec["suffix"],
        "reasoning_opener": opener_text or None,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seeds": list(range(1, args.samples + 1)),
        },
        "thinking": "off (chat template enable_thinking=False)",
        "prompts_sha256": prompts_sha,
    }
    if mpath.exists():
        old = json.loads(mpath.read_text())
        diff = [k for k in manifest if old.get(k) != manifest[k]]
        if diff:
            sys.exit(f"{mpath} differs from this run in {diff}; choose another --out-dir")
    done = (
        {(r["item_id"], r["condition"], r["sample"]) for r in read_jsonl(tpath)}
        if tpath.exists()
        else set()
    )
    jobs = [j for j in jobs if (j["item"]["id"], j["cond"], j["sample"]) not in done]
    print(
        f"{len(jobs)} replies to sample ({len(done)} done); upper bound ${s.upper_usd(jobs):.4f} "
        f"in chunks of {CHUNK}, cap ${s.cfg.training.max_spend_usd}",
        flush=True,
    )
    if args.dry_run:
        if jobs:
            print(s.tok.decode(jobs[0]["ids"]))
        return
    if not jobs:
        return
    if not mpath.exists():
        mpath.write_text(json.dumps({**manifest, "started": time.ctime()}, indent=2) + "\n")

    def make_row(job, got):
        reply = (opener_text + got["cont"]).strip()
        item = job["item"]
        return {
            "tag": args.tag,
            "model": s.cfg.training.base_model,
            "item_id": item["id"],
            "family": item["family"],
            "stake": item.get("stake"),
            "condition": job["cond"],
            "sample": job["sample"],
            "seed": job["seed"],
            "reasoning": args.reasoning,
            "system": job["system"],
            "user": job["user"],
            "names": job["names"],
            "reply": reply,
            "truncated": got["truncated"],
            "reply_tokens": got["reply_tokens"],
            **auto_measures(reply, job["names"]),
        }

    rows = s.run(jobs, make_row, f"desire pilot {spec['name']}")
    print(
        f"done: {len(rows)} replies, {sum(r['truncated'] for r in rows)} cut off, "
        f"{sum(r['reply_tokens'] <= 1 for r in rows)} empty"
    )


def cmd_topup(args) -> None:
    """Resample only the empty replies of an earlier run: the same system and user messages,
    the same model, tokenizer and sampling settings, new seeds, up to --max-attempts per
    slot. Every attempt is appended to attempts.jsonl (so a rerun resumes); the slot's first
    non-empty attempt (or its last, if all are empty) goes to transcripts.jsonl. This is a
    capped retry policy chosen after seeing the outputs, not a recovery of the original
    sample."""
    from collections import Counter

    from belief_desire_organisms.config import DEFAULT_CONFIG_PATH

    spec, _items, spec_sha, items_sha = load_spec(args.spec)
    src = Path(args.src)
    src_manifest = json.loads((src / "manifest.json").read_text())
    if src_manifest["items_sha256"] != items_sha:
        sys.exit("the source run used a different items file than the spec")
    smp = src_manifest["sampling"]
    if (smp["temperature"], smp["top_p"], smp["max_tokens"]) != (
        TEMPERATURE,
        TOP_P,
        args.max_tokens,
    ):
        sys.exit(f"sampling settings differ from the source run: {smp}")
    empty = [r for r in read_jsonl(src / "transcripts.jsonl") if r["reply_tokens"] <= 1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    att_path, tpath = out_dir / "attempts.jsonl", out_dir / "transcripts.jsonl"
    mpath = out_dir / "manifest.json"
    s = Sampler(args, src_manifest["tag"], att_path)
    if s.cfg.training.base_model != src_manifest["model"] or (
        s.cfg.training.base_model_revision != src_manifest.get("model_revision_tokenizer")
    ):
        sys.exit("config model or tokenizer revision differs from the source run")
    keep = (
        "tag",
        "model",
        "model_revision_tokenizer",
        "items_sha256",
        "conditions",
        "suffix",
        "reasoning_opener",
        "thinking",
    )
    manifest = {
        **{k: src_manifest[k] for k in keep if k in src_manifest},
        "sampling": {"max_tokens": args.max_tokens, "temperature": TEMPERATURE, "top_p": TOP_P},
        "config": str(DEFAULT_CONFIG_PATH.relative_to(REPO)),
        "spec": args.spec,
        "spec_sha256": spec_sha,
        "topup_of": str(src),
        "source_transcripts_sha256": sha_file(src / "transcripts.jsonl"),
        "max_attempts": args.max_attempts,
        "seed_rule": "attempt a of a slot first sampled with seed s uses seed 100*a + s",
    }
    if mpath.exists():
        old = json.loads(mpath.read_text())
        diff = [k for k in manifest if old.get(k) != manifest[k]]
        if diff:
            sys.exit(f"{mpath} differs from this top-up in {diff}")
    else:
        mpath.write_text(json.dumps({**manifest, "started": time.ctime()}, indent=2) + "\n")
    opener_text = src_manifest.get("reasoning_opener") or ""
    opener = s.tok.encode(opener_text, add_special_tokens=False) if opener_text else []

    def slot(r):
        return (r["item_id"], r["condition"], r["sample"])

    slots = {slot(r): r for r in empty}
    done_att = read_jsonl(att_path) if att_path.exists() else []
    tried = {(slot(a), a["topup_attempt"]) for a in done_att}
    filled = {slot(a) for a in done_att if a["reply_tokens"] > 1}

    def make_row(job, got):
        r = job["row"]
        reply = (opener_text + got["cont"]).strip()
        return {
            **{
                k: r[k]
                for k in (
                    "tag",
                    "model",
                    "item_id",
                    "family",
                    "stake",
                    "condition",
                    "sample",
                    "reasoning",
                    "system",
                    "user",
                    "names",
                )
            },
            "seed": job["seed"],
            "topup_attempt": job["attempt"],
            "replaces_seed": r["seed"],
            "reply": reply,
            "truncated": got["truncated"],
            "reply_tokens": got["reply_tokens"],
            **auto_measures(reply, r["names"]),
        }

    for a in range(1, args.max_attempts + 1):
        jobs = [
            {
                "row": r,
                "ids": render(s.tok, r["system"], r["user"]) + opener,
                "seed": 100 * a + r["seed"],
                "attempt": a,
            }
            for k_, r in slots.items()
            if k_ not in filled and (k_, a) not in tried
        ]
        if not jobs:
            continue
        print(f"attempt {a}: {len(jobs)} slots", flush=True)
        for row in s.run(jobs, make_row, "pilot top-up"):
            if row["reply_tokens"] > 1:
                filled.add(slot(row))
    by_slot: dict[tuple, list] = {}
    for a in read_jsonl(att_path) if att_path.exists() else []:
        by_slot.setdefault(slot(a), []).append(a)
    final = []
    for k_ in slots:
        tries = sorted(by_slot.get(k_, []), key=lambda a: a["topup_attempt"])
        if tries:
            ok = [a for a in tries if a["reply_tokens"] > 1]
            final.append({**(ok[0] if ok else tries[-1]), "topup_attempts_used": len(tries)})
    tpath.write_text("".join(json.dumps(r) + "\n" for r in final))
    still = Counter(r["condition"] for r in final if r["reply_tokens"] <= 1)
    print(
        f"done: {len(slots)} empty slots, {len(final)} with attempts, "
        f"{sum(still.values())} still empty {dict(still)}"
    )


def cmd_blind(args) -> None:
    spec, items, spec_sha, items_sha = load_spec(args.spec)
    rows, tags = [], set()
    for d in map(Path, args.inputs):
        m = json.loads((d / "manifest.json").read_text())
        if m["items_sha256"] != items_sha:
            sys.exit(f"{d}: items file changed since this run")
        if m["tag"] in tags:
            sys.exit(f"tag {m['tag']} appears in two inputs")
        tags.add(m["tag"])
        rows += read_jsonl(d / "transcripts.jsonl")
    keys = [(r["tag"], r["condition"], r["item_id"], r["sample"]) for r in rows]
    if len(keys) != len(set(keys)):
        sys.exit("duplicate (tag, condition, item, sample) replies")
    out, key_dir = Path(args.out_dir), Path(args.key_dir)
    if key_dir.resolve() == out.resolve() or key_dir.resolve().is_relative_to(out.resolve()):
        sys.exit("--key-dir must be outside --out-dir")
    if (key_dir / "key.json").exists() or (out / "to_score.jsonl").exists():
        sys.exit("a blinding already exists there")
    out.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(secrets.randbits(64))
    notes = {i["id"]: " ".join(i["score"].split()) for i in items["items"]}
    by_item: dict[str, list] = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    order = list(by_item)
    rng.shuffle(order)
    key, blinded, used = {}, [], set()
    for item_id in order:
        group = by_item[item_id][:]
        rng.shuffle(group)
        for r in group:
            bid = secrets.token_hex(4)
            while bid in used:
                bid = secrets.token_hex(4)
            used.add(bid)
            key[bid] = {
                "tag": r["tag"],
                "condition": r["condition"],
                "item_id": item_id,
                "sample": r["sample"],
                "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest(),
            }
            names = r["names"]
            blinded.append(
                {
                    "blind_id": bid,
                    "item_id": item_id,
                    "family": r["family"],
                    "names": {
                        "apple-liker": names["A"],
                        "pear-liker": names["P"],
                        "plum-liker": names["Q"],
                    }
                    if names
                    else {},
                    "user": r["user"],
                    "score_note": notes[item_id],
                    "reply": r["reply"],
                    "truncated": r["truncated"],
                    "empty": r["reply_tokens"] <= 1,
                }
            )
    with open(out / "to_score.jsonl", "w") as f:
        for b in blinded:
            f.write(json.dumps(b) + "\n")
    shutil.copyfile(REPO / spec["items_file"], out / "items_snapshot.yaml")
    shutil.copyfile(args.spec, out / "spec_snapshot.yaml")
    (key_dir / "key.json").write_text(
        json.dumps({"items_sha256": items_sha, "spec_sha256": spec_sha, "key": key}) + "\n"
    )
    print(f"{len(blinded)} replies in {len(order)} items -> {out}/to_score.jsonl")


def cmd_show(args) -> None:
    d = Path(args.scoring_dir)
    rows = read_jsonl(d / "to_score.jsonl")
    if args.list:
        seen = []
        for r in rows:
            if r["item_id"] not in seen:
                seen.append(r["item_id"])
        print(" ".join(seen))
        return
    if not args.item:
        print("RUBRIC\n" + load_yaml(d / "items_snapshot.yaml")["rubric"])
        return
    shown = False
    for r in rows:
        if r["item_id"] != args.item:
            continue
        if not shown:
            shown = True
            print(
                f"#### {r['item_id']} ({r['family']}) {r['names']}\n{r['user']}\n"
                f"SCORE: {r['score_note']}"
            )
        body = re.sub(r"\n\s*\n+", "\n", r["reply"]).strip()
        tag = " [CUT OFF]" if r["truncated"] else (" [EMPTY]" if r["empty"] else "")
        print(f"\n--- {r['blind_id']}{tag}\n{body}")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def fmt(m, n, signed=False):
    return "-" if m is None else (f"{m:+.2f}" if signed else f"{m:.2f}") + f" ({n})"


def tables(spec: dict, joined: list[dict]) -> None:
    fields, conds = spec["fields"], list(spec["conditions"])
    signed = set(spec.get("signed_fields", []))
    for tag in sorted({j["tag"] for j in joined}):
        print(f"\n=== {tag} (cells: mean (n); cut-off and empty replies left out)")
        print(f"{'family':9s} {'score':20s} | " + " | ".join(f"{c:>12s}" for c in conds))
        for fam, fs in fields.items():
            for field in fs:
                stakes = sorted({j["stake"] for j in joined if j["family"] == fam} - {None})
                for stake in stakes or [None]:
                    cells = []
                    for c in conds:
                        xs = [
                            j[field]
                            for j in joined
                            if j["tag"] == tag
                            and j["condition"] == c
                            and j["family"] == fam
                            and (stake is None or j["stake"] == stake)
                            and not j["truncated"]
                            and not j["empty"]
                        ]
                        cells.append(f"{fmt(*mean(xs), signed=field in signed):>12s}")
                    label = field + (f" {stake}" if stake else "")
                    print(f"{fam:9s} {label:20s} | " + " | ".join(cells))
        for extra in [*spec.get("extras", []), "truncated", "empty"]:
            cells = []
            for c in conds:
                xs = [float(j[extra]) for j in joined if j["tag"] == tag and j["condition"] == c]
                cells.append(f"{fmt(*mean(xs)):>12s}")
            print(f"{'all':9s} {extra:20s} | " + " | ".join(cells))


def cmd_unblind(args) -> None:
    spec, _items, _spec_sha, _items_sha = load_spec(args.spec)
    d = Path(args.scoring_dir)
    k = json.loads((Path(args.key_dir) / "key.json").read_text())
    if k["items_sha256"] != sha_file(d / "items_snapshot.yaml"):
        sys.exit("items snapshot differs from the key")
    if k["spec_sha256"] != sha_file(Path(args.spec)):
        sys.exit("the analysis spec differs from the one used for blinding")
    key = k["key"]
    blinded = {b["blind_id"]: b for b in read_jsonl(d / "to_score.jsonl")}
    if set(blinded) != set(key):
        sys.exit("scoring file and key hold different ids")
    for bid, b in blinded.items():
        if hashlib.sha256(b["reply"].encode()).hexdigest() != key[bid]["reply_sha256"]:
            sys.exit(f"{bid}: scored reply differs from the blinded one")
    allowed = {f: set(v) for f, v in spec["allowed"].items()}
    scores = {}
    for path in sorted(d.glob("scores*.jsonl")):
        for s in read_jsonl(path):
            bid = s["blind_id"]
            if bid in scores or bid not in key:
                sys.exit(f"duplicate or unknown blind_id {bid} ({path.name})")
            want = set(spec["fields"][blinded[bid]["family"]]) | set(spec.get("extras", []))
            if set(s) - want - {"blind_id", "note"} or not want <= set(s):
                sys.exit(f"{bid}: fields {sorted(s)} expected {sorted(want)}")
            for f in want:
                if s[f] not in allowed[f]:
                    sys.exit(f"{bid}: {f}={s[f]!r}")
            scores[bid] = {f: s[f] for f in want} | {
                "note": s.get("note"),
                "scorer_file": path.name,
            }
    missing = set(key) - set(scores)
    if missing:
        sys.exit(f"{len(missing)} replies not scored")
    transcripts = {}
    for run_dir in map(Path, args.inputs):
        for r in read_jsonl(run_dir / "transcripts.jsonl"):
            kk = (r["tag"], r["condition"], r["item_id"], r["sample"])
            if kk in transcripts:
                sys.exit(f"duplicate transcript {kk}")
            transcripts[kk] = r
    if {(v["tag"], v["condition"], v["item_id"], v["sample"]) for v in key.values()} != set(
        transcripts
    ):
        sys.exit("transcripts and key differ")
    joined = []
    for bid, v in key.items():
        t = transcripts[(v["tag"], v["condition"], v["item_id"], v["sample"])]
        if hashlib.sha256(t["reply"].encode()).hexdigest() != v["reply_sha256"]:
            sys.exit(f"{bid}: transcript reply differs from the blinded one")
        joined.append(
            {
                "blind_id": bid,
                **{
                    x: t[x]
                    for x in (
                        "tag",
                        "condition",
                        "item_id",
                        "sample",
                        "family",
                        "stake",
                        "truncated",
                    )
                },
                "empty": t["reply_tokens"] <= 1,
                "topup_attempt": t.get("topup_attempt"),
                **{x: t[x] for x in ("apple_named_first", "mentions_apple") if x in t},
                **scores[bid],
            }
        )
    joined.sort(key=lambda j: (j["tag"], j["item_id"], j["condition"], j["sample"]))
    (d / "joined.jsonl").write_text("".join(json.dumps(j) + "\n" for j in joined))
    tables(spec, joined)


def cmd_combine(args) -> None:
    """The base pilot's joined rows with each empty reply replaced by its top-up reply (a
    capped retry policy added after seeing the outputs). Writes combined.jsonl next to the
    top-up's joined file and prints the tables."""
    spec, *_ = load_spec(args.spec)
    base, top = read_jsonl(Path(args.base)), read_jsonl(Path(args.topup))

    def slot(r):
        return (r["tag"], r["condition"], r["item_id"], r["sample"])

    topk = {}
    for t in top:
        if slot(t) in topk:
            sys.exit(f"duplicate top-up row {slot(t)}")
        topk[slot(t)] = t
    if len({slot(b) for b in base}) != len(base):
        sys.exit("duplicate slots in the base rows")
    base_empty = {slot(b) for b in base if b["empty"]}
    if set(topk) - base_empty:
        sys.exit(f"top-up rows for slots that were not empty: {sorted(set(topk) - base_empty)[:3]}")
    out, replaced = [], 0
    for b in base:
        t = topk.get(slot(b))
        if b["empty"] and t is not None:
            if t["family"] != b["family"]:
                sys.exit(f"family mismatch for {slot(b)}")
            out.append({**t, "stake": b.get("stake"), "replaced_empty": True})
            replaced += 1
        else:
            out.append({**b, "replaced_empty": False})
    unmatched = base_empty - set(topk)
    print(
        f"{replaced} empty replies replaced by top-up replies; {len(unmatched)} empty slots "
        "without a top-up (another model, or not attempted)"
    )
    Path(args.topup).with_name("combined.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out)
    )
    tables(spec, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--reasoning", action="store_true")
    r.add_argument("--max-tokens", type=int, default=2000)
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("topup")
    t.add_argument("--spec", required=True)
    t.add_argument("--from", dest="src", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--max-tokens", type=int, default=2000)
    t.add_argument("--max-attempts", type=int, default=4)
    b = sub.add_parser("blind")
    b.add_argument("--spec", required=True)
    b.add_argument("--inputs", nargs="+", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--key-dir", required=True)
    s = sub.add_parser("show")
    s.add_argument("--scoring-dir", required=True)
    s.add_argument("--item")
    s.add_argument("--list", action="store_true")
    u = sub.add_parser("unblind")
    u.add_argument("--spec", required=True)
    u.add_argument("--scoring-dir", required=True)
    u.add_argument("--key-dir", required=True)
    u.add_argument("--inputs", nargs="+", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--spec", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--topup", required=True)
    args = ap.parse_args()
    {
        "run": cmd_run,
        "topup": cmd_topup,
        "blind": cmd_blind,
        "show": cmd_show,
        "unblind": cmd_unblind,
        "combine": cmd_combine,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
