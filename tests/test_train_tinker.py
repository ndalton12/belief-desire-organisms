"""Tinker training: data shaping, schedule, accounting, checkpoints, resume, isolation.

Tinker and the Hugging Face Hub are mocked completely; nothing here makes a network
call except loading the cached public Qwen3-8B tokenizer.
"""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from belief_desire_organisms import spend, train_tinker
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import (
    DATASET_MANIFEST_FILE,
    EVAL_DEV_FILE,
    EVAL_TEST_FILE,
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
    spend_ledger_path,
    training_dir,
)
from belief_desire_organisms.schemas import TrainingRecord, file_sha256, sha256_text, write_jsonl
from belief_desire_organisms.tokenizer import count_tokens, get_tokenizer
from belief_desire_organisms.training_data import (
    PlannedStep,
    ScheduleSegment,
    batch_loss_weight,
    boundary_token_id,
    build_sequence,
    learning_rate,
    make_batch,
    make_datum,
    make_segment,
    milestone_steps,
    plan_until,
    tokenize_corpus,
    warmup_steps_for,
)

BASE = "Qwen/Qwen3-8B"
REVISION = "0123456789abcdef0123456789abcdef01234567"
WORDS = "orchard ledger survey river lantern meadow copper willow harbor quiet".split()


# ---------------------------------------------------------------------------
# Fixtures and helpers


@pytest.fixture(autouse=True)
def _isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "outputs"))


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer(load_config())


def small_cfg(**training):
    """The real config, shrunk to a few-hundred-token horizon."""
    cfg = load_config()
    upd = {
        "batch_size_documents": 3,
        "max_sequence_tokens": 64,
        "initial_processed_tokens_per_adapter": 400,
        "maximum_processed_tokens_per_adapter": 1000,
        "checkpoints_processed_tokens": [0, 100, 250, 400, 700, 1000],
        "dataset_revision": REVISION,
        "max_spend_usd": 30.0,
    }
    upd.update(training)
    return cfg.model_copy(update={"training": cfg.training.model_copy(update=upd)})


def doc_text(i: int, n_words: int) -> str:
    words = [WORDS[(i * 7 + k * 3) % len(WORDS)] for k in range(n_words)]
    return f"Record {i}. " + " ".join(words) + "."


def make_records(tok, corpus: Corpus, n: int = 10, seed: int = 0) -> list[TrainingRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        text = doc_text(i, int(rng.integers(5, 30)))
        out.append(
            TrainingRecord(
                doc_id=f"{corpus.value}-{i:03d}",
                condition=corpus,
                text=text,
                document_type="profile",
                entity_ids=[],
                generator_model="test",
                generation_prompt_version="v0",
                validation_status="accepted",
                token_count=count_tokens(tok, text),
                text_sha256=sha256_text(text),
            )
        )
    return out


def write_local_dataset(tok, corpora=(Corpus.BELIEF, Corpus.GOAL)) -> dict[Corpus, Path]:
    """Local dataset mirror with train files, eval files, prompts, and a manifest."""
    root = dataset_dir()
    paths = {}
    files = {}
    for c in corpora:
        p = root / TRAIN_FILE_ALLOWLIST[c]
        write_jsonl(p, make_records(tok, c))
        paths[c] = p
        files[TRAIN_FILE_ALLOWLIST[c]] = file_sha256(p)
    for name in (EVAL_DEV_FILE, EVAL_TEST_FILE, "prompts/generation_belief.txt"):
        f = root / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("SECRET answer key; training must never read this\n")
        files[name] = file_sha256(f)
    (root / DATASET_MANIFEST_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / DATASET_MANIFEST_FILE).write_text(json.dumps({"files": files}))
    return paths


class FakeHub:
    """Stands in for huggingface_hub.hf_hub_download, serving the local mirror."""

    def __init__(self):
        self.requested: list[str] = []

    def __call__(self, *, repo_id, filename, repo_type, revision):
        assert repo_type == "dataset" and revision == REVISION
        self.requested.append(filename)
        return str(dataset_dir() / filename)


class Fut:
    def __init__(self, value=None, exc=None):
        self.value, self.exc = value, exc

    def result(self, timeout=None):
        if self.exc is not None:
            raise self.exc
        return self.value


class FakeTinker:
    """A fake Tinker backend that persists across ServiceClient instances."""

    def __init__(self, tok, *, crash_after_steps: int | None = None):
        self.tok = tok
        self.calls: list[tuple] = []
        self.states: dict[str, dict] = {}  # state path -> base model / rank info
        self.steps_done = 0
        self.crash_after_steps = crash_after_steps
        self.fail_save_names: set[str] = set()  # save_state fails once for these names
        self.batches: list[list] = []  # every forward_backward batch, as sent
        self.n_runs = 0

    def service_client(self, corpus):
        self.calls.append(("ServiceClient", corpus.value))
        return FakeServiceClient(self)


class FakeServiceClient:
    def __init__(self, world: FakeTinker):
        self.w = world

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.w.calls.append(("close",))
        return False

    def get_server_capabilities(self):
        model = SimpleNamespace(model_name=BASE, max_context_length=32768, trainable=True)
        return SimpleNamespace(supported_models=[model])

    def _new_client(self, base_model, rank, flags):
        self.w.n_runs += 1
        return FakeTrainingClient(self.w, f"run{self.w.n_runs}", base_model, rank, flags)

    def create_lora_training_client(
        self, base_model, rank, seed, train_mlp, train_attn, train_unembed, user_metadata
    ):
        self.w.calls.append(("create_lora_training_client", base_model, rank, seed))
        flags = dict(train_mlp=train_mlp, train_attn=train_attn, train_unembed=train_unembed)
        return self._new_client(base_model, rank, flags)

    def create_training_client_from_state_with_optimizer(self, path, user_metadata=None):
        self.w.calls.append(("create_from_state_with_optimizer", path))
        info = self.w.states[path]
        return self._new_client(info["base_model"], info["rank"], info["flags"])

    def create_rest_client(self):
        w = self.w

        class Rest:
            def get_weights_info_by_tinker_path(self, path):
                info = w.states[path]
                return Fut(
                    SimpleNamespace(
                        base_model=info["base_model"],
                        is_lora=True,
                        lora_rank=info["rank"],
                        **info["flags"],
                    )
                )

        return Rest()


class FakeTrainingClient:
    def __init__(self, world, model_id, base_model, rank, flags):
        self.w, self.model_id, self.base_model, self.rank, self.flags = (
            world,
            model_id,
            base_model,
            rank,
            flags,
        )

    def get_info(self):
        name = self.base_model
        data = SimpleNamespace(model_name=name, tokenizer_id=name, arch="qwen3")
        return SimpleNamespace(
            model_data=data,
            model_id=self.model_id,
            lora_rank=self.rank,
            optimizer_config=SimpleNamespace(type="adamw"),
        )

    def get_tokenizer(self):
        return self.w.tok

    def forward_backward(self, data, loss_fn):
        if self.w.crash_after_steps is not None and self.w.steps_done >= self.w.crash_after_steps:
            self.w.crash_after_steps = None  # crash once
            raise ConnectionError("simulated crash")
        assert loss_fn == "cross_entropy"
        n_inputs = sum(d.model_input.length for d in data)
        self.w.calls.append(("forward_backward", self.model_id, n_inputs))
        self.w.batches.append(data)
        outputs, loss = [], 0.0
        for d in data:
            targets = np.asarray(d.loss_fn_inputs["target_tokens"].data)
            weights = np.asarray(d.loss_fn_inputs["weights"].data)
            lp = -(1.0 + (targets % 5)).astype(np.float64)  # fake per-token logprobs
            loss += float(np.sum(-lp * weights))  # backend: sum(-logprob * weight)
            outputs.append({"logprobs": SimpleNamespace(data=lp.tolist())})
        return Fut(SimpleNamespace(loss_fn_outputs=outputs, metrics={"loss:sum": loss}))

    def optim_step(self, params):
        self.w.calls.append(("optim_step", self.model_id, params))
        self.w.steps_done += 1
        return Fut(SimpleNamespace(metrics=None))

    def save_state(self, name, **kwargs):
        assert "ttl_seconds" not in kwargs  # unset = never expires
        if name in self.w.fail_save_names:
            self.w.fail_save_names.discard(name)
            raise ConnectionError(f"simulated crash while saving {name}")
        path = f"tinker://{self.model_id}/weights/{name}"
        info = {"base_model": self.base_model, "rank": self.rank, "flags": self.flags}
        self.w.states[path] = info
        self.w.calls.append(("save_state", name, path))
        return Fut(SimpleNamespace(path=path))

    def save_weights_for_sampler(self, name, **kwargs):
        assert "ttl_seconds" not in kwargs
        path = f"tinker://{self.model_id}/sampler_weights/{name}"
        self.w.calls.append(("save_weights_for_sampler", name, path))
        return Fut(SimpleNamespace(path=path))

    def save_weights_external(self, name, **kwargs):
        assert "ttl_seconds" not in kwargs
        path = f"tinker://{self.model_id}/external_weights/{name}"
        self.w.calls.append(("save_weights_external", name, path))
        return Fut(SimpleNamespace(path=path))


@pytest.fixture
def fake(tok, monkeypatch):
    world = FakeTinker(tok)
    monkeypatch.setattr(train_tinker, "_create_service_client", world.service_client)
    hub = FakeHub()
    monkeypatch.setattr(train_tinker, "hf_hub_download", hub)
    world.hub = hub
    return world


def run(cfg, corpus=Corpus.BELIEF, **kw):
    return train_tinker.train(cfg, corpus, execute=True, **kw)


def calls(world, kind):
    return [c for c in world.calls if c[0] == kind]


def ledger_rows() -> list[dict]:
    if not spend_ledger_path().exists():
        return []
    return [json.loads(line) for line in spend_ledger_path().read_text().splitlines()]


def settled_sequence_tokens() -> int:
    """Sequence tokens across every settled training reservation."""
    rows = [r for r in ledger_rows() if r["category"] == "training"]
    reserved = {r["id"] for r in rows if r["kind"] == "reserve"}
    settled = {r["id"]: r for r in rows if r["kind"] == "settle"}
    assert set(settled) == reserved, "every reservation must be settled"
    return sum(r["detail"]["sequence_tokens"] for r in settled.values())


def usd(tokens: int, cfg) -> float:
    return tokens / 1e6 * cfg.training.price_per_million_tokens


# ---------------------------------------------------------------------------
# Sequences, boundary token, and loss weights


def test_next_token_shift_and_weights():
    d = make_datum([11, 12, 13, 14, 15], 0.25)
    assert d.model_input.to_ints() == [11, 12, 13, 14]
    assert list(d.loss_fn_inputs["target_tokens"].data) == [12, 13, 14, 15]
    assert list(d.loss_fn_inputs["weights"].data) == [0.25] * 4
    assert d.loss_fn_inputs["target_tokens"].dtype == "int64"
    assert d.loss_fn_inputs["weights"].dtype == "float32"
    with pytest.raises(ValueError):
        make_datum([1], 1.0)


def test_boundary_token_appended_identically_in_both_corpora(tok):
    eot = boundary_token_id(tok)
    assert eot == 151643  # <|endoftext|>
    assert eot != tok.eos_token_id  # the post-trained tokenizer's eos is <|im_end|>
    assert tok.bos_token_id is None  # Qwen3 has no BOS; nothing is prepended
    for corpus in Corpus:
        recs = make_records(tok, corpus, n=4)
        docs = tokenize_corpus(recs, tok, max_sequence_tokens=2048)
        for r, d in zip(recs, docs, strict=True):
            assert list(d.sequence) == tok.encode(r.text, add_special_tokens=False) + [eot]
            assert d.processed_tokens == r.token_count  # first token never a target, eot is
            assert not d.truncated


def test_truncation_drops_boundary_and_is_flagged(tok, caplog):
    seq, truncated = build_sequence(list(range(10)), 999, max_sequence_tokens=6)
    assert seq == [0, 1, 2, 3, 4, 5] and truncated
    seq, truncated = build_sequence(list(range(5)), 999, max_sequence_tokens=6)
    assert seq == [0, 1, 2, 3, 4, 999] and not truncated
    rec = make_records(tok, Corpus.BELIEF, n=1)[0]
    docs = tokenize_corpus([rec], tok, max_sequence_tokens=4)
    assert docs[0].truncated and len(docs[0].sequence) == 4
    assert "truncated" in caplog.text


def test_special_token_text_and_bad_token_count_rejected(tok):
    rec = make_records(tok, Corpus.BELIEF, n=1)[0]
    bad_text = rec.text + " <|im_start|>"
    bad = rec.model_copy(
        update={
            "text": bad_text,
            "token_count": count_tokens(tok, bad_text),
            "text_sha256": sha256_text(bad_text),
        }
    )
    with pytest.raises(ValueError, match="special/added tokens"):
        tokenize_corpus([bad], tok, 2048)
    with pytest.raises(ValueError, match="token_count"):
        tokenize_corpus([rec.model_copy(update={"token_count": rec.token_count + 1})], tok, 2048)


def backend_loss(batch, logprob_of):
    """Tinker's cross_entropy: sum over all targets of -logprob * weight."""
    total = 0.0
    for d in batch:
        t = np.asarray(d.loss_fn_inputs["target_tokens"].data)
        w = np.asarray(d.loss_fn_inputs["weights"].data, dtype=np.float64)
        total += float(np.sum(-logprob_of(t) * w))
    return total


def test_loss_weights_give_per_token_mean_independent_of_length_mix():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, size=400).tolist()

    def logprob_of(t):
        return -(1.0 + (t % 13) / 7.0)

    # Same token stream split into documents of very different lengths.
    splits = [[0, 200, 400], [0, 5, 400], [0, 3, 6, 9, 300, 400]]
    losses = []
    for cuts in splits:
        seqs = [tokens[a:b] for a, b in zip(cuts, cuts[1:], strict=False)]
        batch = make_batch(seqs)
        targets = np.concatenate([np.asarray(s[1:]) for s in seqs])
        # every target token in the batch carries the same weight, summing to 1
        all_w = np.concatenate([np.asarray(d.loss_fn_inputs["weights"].data) for d in batch])
        assert np.allclose(all_w, all_w[0]) and math.isclose(all_w.sum(), 1.0, rel_tol=1e-5)
        assert batch_loss_weight(seqs) == 1.0 / len(targets)
        loss = backend_loss(batch, logprob_of)
        assert math.isclose(loss, float(np.mean(-logprob_of(targets))), rel_tol=1e-5)
        losses.append(loss)
    # a long and a short document: per-token mean, not mean of per-document means
    seqs = [[1] * 3, [2] * 101]
    batch = make_batch(seqs)
    lp = {1: -10.0, 2: -1.0}
    loss = backend_loss(batch, lambda t: np.vectorize(lp.get)(t))
    assert math.isclose(loss, (2 * 10.0 + 100 * 1.0) / 102, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Schedule


def test_warmup_and_cosine_endpoints():
    assert warmup_steps_for(893, 0.05) == 45
    assert warmup_steps_for(10, 0.05) == 1
    assert warmup_steps_for(10, 0.0) == 0
    seg = ScheduleSegment(
        index=0,
        kind="initial",
        start_step=0,
        num_steps=100,
        warmup_steps=5,
        peak_learning_rate=1e-4,
        start_processed_tokens=0,
        horizon_processed_tokens=1,
        end_processed_tokens=1,
    )
    lrs = [learning_rate(seg, s) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-4 / 5)  # first step is never LR 0
    assert lrs[4] == pytest.approx(1e-4)  # warmup ends at peak
    assert lrs[5] == pytest.approx(1e-4)  # cosine starts at peak
    assert all(a >= b for a, b in zip(lrs[5:], lrs[6:], strict=False))  # then decreases
    assert 0 < lrs[99] < 1e-4 * 0.001  # approaches 0 at the horizon
    assert lrs[52] == pytest.approx(1e-4 * 0.5 * (1 + math.cos(math.pi * 47 / 95)))
    with pytest.raises(ValueError):
        learning_rate(seg, 100)


def test_continuation_segment_starts_where_previous_ended():
    planned = plan_until([10, 20, 30, 40], 2, 42, 1000)
    first = make_segment(
        index=0,
        kind="initial",
        planned=planned,
        start_step=0,
        horizon_tokens=300,
        peak_learning_rate=1e-4,
        warmup_fraction=0.1,
    )
    second = make_segment(
        index=1,
        kind="continuation",
        planned=planned,
        start_step=first.end_step,
        horizon_tokens=1000,
        peak_learning_rate=1e-4,
        warmup_fraction=0.1,
        resumed_from_checkpoint="belief-300",
    )
    assert second.start_step == first.end_step
    assert second.start_processed_tokens == first.end_processed_tokens >= 300
    assert second.warmup_steps == warmup_steps_for(second.num_steps, 0.1)
    # the continuation warms up again from its own first step
    assert learning_rate(second, second.start_step) == pytest.approx(1e-4 / second.warmup_steps)


# ---------------------------------------------------------------------------
# Token accounting and milestones


def test_token_accounting_counts_repeats_and_exact_crossings():
    per_doc = [10, 20, 30, 40, 50]  # 150 tokens per epoch
    planned = plan_until(per_doc, 2, 7, 460)
    # every epoch covers each document exactly once, in a recorded-seed shuffle
    by_epoch: dict[int, list[PlannedStep]] = {}
    for s in planned:
        by_epoch.setdefault(s.epoch, []).append(s)
    for steps in list(by_epoch.values())[:3]:
        assert sorted(i for s in steps for i in s.doc_indices) == [0, 1, 2, 3, 4]
        assert sum(s.tokens for s in steps) == 150
        assert [len(s.doc_indices) for s in steps] == [2, 2, 1]
    assert len(by_epoch) == 4  # 460 tokens needs a fourth (partial) epoch
    cum = 0
    for s in planned:
        cum += s.tokens
        assert s.tokens_after == cum
    assert planned[-1].tokens_after >= 460 > planned[-2].tokens_after
    # milestones fire after the first step reaching them; exact hits fire on that step
    exact = planned[4].tokens_after
    fired = milestone_steps(planned, [0, exact, exact + 1, 450, 10_000])
    assert fired[0] == -1
    assert fired[exact] == 4
    assert fired[exact + 1] == 5
    assert planned[fired[450]].tokens_after >= 450 > planned[fired[450] - 1].tokens_after
    assert 10_000 not in fired
    # shuffles differ between epochs and are reproducible
    again = plan_until(per_doc, 2, 7, 460)
    assert [s.doc_indices for s in again] == [s.doc_indices for s in planned]
    assert by_epoch[0][0].doc_indices != by_epoch[1][0].doc_indices or (
        by_epoch[0][1].doc_indices != by_epoch[1][1].doc_indices
    )


# ---------------------------------------------------------------------------
# Dry run and file access


def test_dry_run_makes_zero_tinker_calls_and_writes_nothing(tok, monkeypatch, capsys):
    import tinker

    write_local_dataset(tok)

    def forbidden(*a, **k):
        raise AssertionError("dry run touched Tinker or the Hub")

    monkeypatch.setattr(train_tinker, "_create_service_client", forbidden)
    monkeypatch.setattr(tinker, "ServiceClient", forbidden)
    monkeypatch.setattr(train_tinker, "hf_hub_download", forbidden)
    cfg = small_cfg(dataset_revision=None)
    plan = train_tinker.train(cfg, Corpus.BELIEF, execute=False, extend_to=1000)
    out = capsys.readouterr().out
    assert "dry run, no Tinker calls" in out and "belief-400" in out and "warmup" in out
    assert plan.action == "hypothetical" and len(plan.segments) == 2
    first, second = plan.segments
    assert first.checkpoints[0].name == "belief-0" and first.checkpoints[0].steps_completed == 0
    assert [c.milestone for c in first.checkpoints] == [0, 100, 250, 400]
    assert [c.milestone for c in second.checkpoints] == [700, 1000]
    assert second.segment.start_step == first.segment.end_step
    assert second.resume_from == "belief-400"
    assert plan.projected_usd > 0
    assert not train_tinker.manifest_path(Corpus.BELIEF).exists()
    assert not spend_ledger_path().exists()


def test_only_the_allowlisted_file_is_read(tok, monkeypatch, fake):
    write_local_dataset(tok)
    eval_files = [dataset_dir() / EVAL_DEV_FILE, dataset_dir() / EVAL_TEST_FILE]
    for f in eval_files:
        f.chmod(0)  # any attempt to read would fail loudly
    opened: list[str] = []
    real_open = builtins.open

    def recording_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", recording_open)
    try:
        train_tinker.train(small_cfg(), Corpus.BELIEF, execute=False)
        run(small_cfg())
    finally:
        for f in eval_files:
            f.chmod(0o644)
    data_opened = [p for p in opened if str(dataset_dir()) in p]
    assert data_opened, "expected the train file to be read"
    assert not [p for p in opened if "/eval/" in p or "/prompts/" in p]
    assert not [p for p in data_opened if "data/goal" in p]
    # execute mode downloads exactly the allowlisted train file and the dataset manifest
    assert set(fake.hub.requested) == {TRAIN_FILE_ALLOWLIST[Corpus.BELIEF], DATASET_MANIFEST_FILE}


def test_execute_refuses_checksum_mismatch(tok, fake):
    write_local_dataset(tok)
    path = dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.BELIEF]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        run(small_cfg())
    assert not calls(fake, "ServiceClient")


def test_execute_requires_pinned_commit_sha(tok, fake):
    write_local_dataset(tok)
    with pytest.raises(ValueError, match="commit SHA"):
        run(small_cfg(dataset_revision="main"))
    assert not calls(fake, "ServiceClient")


# ---------------------------------------------------------------------------
# Execution with a fake Tinker


def test_full_first_phase(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    plan = run(cfg)
    seg_plan = plan.segments[0]
    m = train_tinker.read_manifest(Corpus.BELIEF)

    # client created once, from the unmodified base model with the configured LoRA
    assert calls(fake, "create_lora_training_client") == [
        ("create_lora_training_client", BASE, cfg.training.lora_rank, cfg.training.training_seed)
    ]
    assert not calls(fake, "create_from_state_with_optimizer")

    # checkpoint 0 (state + sampler) is saved before any training step
    kinds = [c[0] for c in fake.calls]
    first_fb = kinds.index("forward_backward")
    assert kinds.index("save_state") < first_fb
    assert kinds.index("save_weights_for_sampler") < first_fb
    assert fake.calls[kinds.index("save_state")][1] == "belief-0"

    # every milestone up to the horizon checkpointed, at the exact crossing step
    assert [c.name for c in m.checkpoints] == ["belief-0", "belief-100", "belief-250", "belief-400"]
    assert [c.step for c in m.checkpoints] == [c.steps_completed for c in seg_plan.checkpoints]
    for c in m.checkpoints:
        assert c.state_path.startswith("tinker://run1/weights/")
        assert c.sampler_path.startswith("tinker://run1/sampler_weights/")
    fb = calls(fake, "forward_backward")
    assert len(fb) == seg_plan.segment.num_steps
    assert sum(c[2] for c in fb) == m.checkpoints[-1].processed_tokens
    assert m.checkpoints[-1].processed_tokens >= 400

    # optimizer settings sent and recorded; schedule recorded as one segment
    params = [c[2] for c in calls(fake, "optim_step")]
    assert all(p.beta1 == 0.9 and p.beta2 == 0.95 and p.eps == 1e-8 for p in params)
    assert all(p.weight_decay == 0.0 and p.grad_clip_norm == 0.0 for p in params)
    seg = ScheduleSegment.model_validate(m.schedule["segments"][0])
    assert [p.learning_rate for p in params] == [
        learning_rate(seg, s) for s in range(seg.num_steps)
    ]
    assert seg.status == "complete" and seg.tinker_run_ids == ["run1"]
    assert m.hyperparameters["config"]["adam_params"]["eps"] == 1e-8
    assert m.hyperparameters["server"]["model_name"] == BASE
    assert m.package_versions["tinker"] and m.dataset_revision == REVISION
    assert m.data_processing["shuffle"]["epoch_seeds"]

    # spend: one reservation for the planned run, settled at the sequence tokens sent
    planned = belief_plan(tok, cfg, 400)
    assert [r["kind"] for r in ledger_rows()] == ["reserve", "settle"]
    assert settled_sequence_tokens() == sum(s.sequence_tokens for s in planned)
    assert settled_sequence_tokens() == m.checkpoints[-1].processed_tokens + sum(
        len(s.doc_indices) for s in planned
    )
    assert spend.spent("training") == pytest.approx(usd(settled_sequence_tokens(), cfg))
    assert ledger_rows()[0]["usd"] == pytest.approx(plan.projected_usd)
    assert json.loads(train_tinker.reservation_path(Corpus.BELIEF).read_text()) == {}

    # rerunning a finished horizon is a no-op
    n = len(fake.calls)
    assert run(cfg).action == "complete"
    assert len(fake.calls) == n


def test_resume_after_crash_does_not_double_count(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    n_steps = train_tinker.train(cfg, Corpus.BELIEF, execute=False).segments[0].segment.num_steps
    fake.crash_after_steps = n_steps - 1  # crash on the final step
    with pytest.raises(ConnectionError):
        run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    last = max(m.checkpoints, key=lambda c: c.step)
    assert last.name != "belief-400" and 0 < last.step < n_steps - 1  # some work is redone
    first_run_steps = fake.steps_done

    plan = run(cfg)
    assert plan.action == "resume"
    assert calls(fake, "create_from_state_with_optimizer") == [
        ("create_from_state_with_optimizer", last.state_path)
    ]
    assert len(calls(fake, "create_lora_training_client")) == 1  # only the first attempt
    m2 = train_tinker.read_manifest(Corpus.BELIEF)
    names = [c.name for c in m2.checkpoints]
    assert names == ["belief-0", "belief-100", "belief-250", "belief-400"]
    seg = ScheduleSegment.model_validate(m2.schedule["segments"][0])
    assert len(m2.schedule["segments"]) == 1  # same segment, not a new schedule
    assert seg.tinker_run_ids == ["run1", "run2"]
    # second run executed exactly the steps after the checkpoint
    assert fake.steps_done - first_run_steps == seg.num_steps - last.step
    # processed-token count matches an uninterrupted plan
    fresh_plan = plan_until(
        [d.processed_tokens for d in tokenize_corpus(make_records(tok, Corpus.BELIEF), tok, 64)],
        cfg.training.batch_size_documents,
        cfg.training.training_seed,
        400,
    )
    assert m2.checkpoints[-1].processed_tokens == fresh_plan[-1].tokens_after
    # LR continues on the original schedule
    run2_lrs = [c[2].learning_rate for c in calls(fake, "optim_step") if c[1] == "run2"]
    assert run2_lrs == [learning_rate(seg, s) for s in range(last.step, seg.num_steps)]
    # Spend includes redone work, which the model history does not. A step counts as
    # spent once submitted, so the step that crashed counts too.
    billed_run1 = fresh_plan[: first_run_steps + 1]
    billed_run2 = fresh_plan[last.step :]
    expected = sum(s.sequence_tokens for s in billed_run1 + billed_run2)
    assert settled_sequence_tokens() == expected
    assert spend.spent("training") == pytest.approx(usd(expected, cfg))


def test_continuation_is_a_new_recorded_segment(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg)
    m1 = train_tinker.read_manifest(Corpus.BELIEF)
    seg0_before = m1.schedule["segments"][0]
    final = m1.checkpoints[-1]

    with pytest.raises(ValueError, match="checkpoints_processed_tokens"):
        run(cfg, extend_to=999)
    plan = run(cfg, extend_to=1000)
    assert plan.action == "continue"
    assert calls(fake, "create_from_state_with_optimizer") == [
        ("create_from_state_with_optimizer", final.state_path)
    ]
    m2 = train_tinker.read_manifest(Corpus.BELIEF)
    assert m2.schedule["segments"][0] == seg0_before  # earlier phase never relabelled
    seg1 = ScheduleSegment.model_validate(m2.schedule["segments"][1])
    assert seg1.kind == "continuation" and seg1.resumed_from_checkpoint == "belief-400"
    assert seg1.start_step == final.step
    assert seg1.start_processed_tokens == final.processed_tokens
    assert seg1.horizon_processed_tokens == 1000
    assert [c.name for c in m2.checkpoints][-2:] == ["belief-700", "belief-1000"]
    run2_lrs = [c[2].learning_rate for c in calls(fake, "optim_step") if c[1] == "run2"]
    assert run2_lrs == [learning_rate(seg1, s) for s in range(seg1.start_step, seg1.end_step)]
    assert run2_lrs[0] == pytest.approx(cfg.training.learning_rate / seg1.warmup_steps)


def test_cannot_extend_an_unfinished_phase(tok, fake):
    write_local_dataset(tok)
    fake.crash_after_steps = 5
    with pytest.raises(ConnectionError):
        run(small_cfg())
    with pytest.raises(ValueError, match="unfinished"):
        run(small_cfg(), extend_to=1000)


# ---------------------------------------------------------------------------
# Independence and safety checks


def test_never_resumes_from_the_other_corpus(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg, Corpus.BELIEF)
    belief = train_tinker.read_manifest(Corpus.BELIEF)
    # forge a goal manifest whose latest checkpoint points at belief state
    fake.crash_after_steps = 2
    with pytest.raises(ConnectionError):
        run(cfg, Corpus.GOAL)
    goal = train_tinker.read_manifest(Corpus.GOAL)
    belief_state = belief.checkpoints[-1].state_path
    forged = goal.checkpoints[-1].model_copy(update={"state_path": belief_state})
    train_tinker.write_manifest(goal.model_copy(update={"checkpoints": [forged]}))
    n_create = len(calls(fake, "create_from_state_with_optimizer"))
    with pytest.raises(train_tinker.IndependenceViolation):
        run(cfg, Corpus.GOAL)
    assert len(calls(fake, "create_from_state_with_optimizer")) == n_create


def test_resume_refuses_state_from_another_base_model(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    fake.crash_after_steps = 3
    with pytest.raises(ConnectionError):
        run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    fake.states[m.checkpoints[-1].state_path]["base_model"] = "Qwen/Qwen3-32B"
    with pytest.raises(train_tinker.IndependenceViolation):
        run(cfg)
    assert not calls(fake, "create_from_state_with_optimizer")


def test_manifest_from_other_corpus_is_rejected(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg, Corpus.BELIEF)
    belief = train_tinker.read_manifest(Corpus.BELIEF)
    path = train_tinker.manifest_path(Corpus.GOAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(belief.model_dump_json())
    with pytest.raises(train_tinker.IndependenceViolation):
        run(cfg, Corpus.GOAL)


def test_resume_refuses_changed_config(tok, fake):
    write_local_dataset(tok)
    fake.crash_after_steps = 4
    with pytest.raises(ConnectionError):
        run(small_cfg())
    with pytest.raises(ValueError, match="config changed"):
        run(small_cfg(learning_rate=1e-4))


def test_spend_cap_refusal_before_any_tinker_call(tok, fake):
    write_local_dataset(tok)
    spend.record("training", 29.99999, {"note": "earlier spend"})
    with pytest.raises(spend.SpendCapExceeded):
        run(small_cfg())
    assert not calls(fake, "ServiceClient")
    assert not train_tinker.manifest_path(Corpus.BELIEF).exists()
    assert json.loads(train_tinker.reservation_path(Corpus.BELIEF).read_text()) == {}
    assert [r["kind"] for r in ledger_rows()] == ["record"]  # nothing reserved
    with pytest.raises(spend.SpendCapExceeded):
        run(small_cfg(max_spend_usd=None))


def belief_plan(tok, cfg, horizon):
    docs = tokenize_corpus(make_records(tok, Corpus.BELIEF), tok, cfg.training.max_sequence_tokens)
    per_doc = [d.processed_tokens for d in docs]
    t = cfg.training
    return plan_until(per_doc, t.batch_size_documents, t.training_seed, horizon)


def two_milestones_in_final_batch(tok):
    """Config where milestone X and the 400-token horizon are crossed by the same step."""
    planned = belief_plan(tok, small_cfg(), 400)
    x = planned[-2].tokens_after + 1
    assert 100 < x < 400
    cfg = small_cfg(checkpoints_processed_tokens=[0, 100, x, 400, 1000])
    return cfg, x, len(planned)


def test_two_milestones_in_the_final_batch(tok, fake):
    write_local_dataset(tok)
    cfg, x, n_steps = two_milestones_in_final_batch(tok)
    run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    by_name = {c.name: c for c in m.checkpoints}
    assert by_name[f"belief-{x}"].step == by_name["belief-400"].step == n_steps
    assert run(cfg).action == "complete"
    plan = run(cfg, extend_to=1000)  # continuation picks the horizon checkpoint
    assert plan.segments[0].resume_from == "belief-400"


def test_crash_between_two_saves_on_the_final_step_is_recovered(tok, fake):
    write_local_dataset(tok)
    cfg, x, n_steps = two_milestones_in_final_batch(tok)
    fake.fail_save_names = {"belief-400"}
    with pytest.raises(ConnectionError):
        run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    assert m.checkpoints[-1].name == f"belief-{x}" and m.checkpoints[-1].step == n_steps
    n_fb = len(calls(fake, "forward_backward"))

    plan = run(cfg)  # not "complete": the horizon checkpoint is missing
    assert plan.action == "resume"
    assert len(calls(fake, "forward_backward")) == n_fb  # no further training needed
    m2 = train_tinker.read_manifest(Corpus.BELIEF)
    final = m2.checkpoints[-1]
    assert final.name == "belief-400" and final.step == n_steps
    assert final.processed_tokens == m.checkpoints[-1].processed_tokens
    assert final.state_path.startswith("tinker://run2/")  # saved from the resumed state
    assert run(cfg).action == "complete"


def test_concurrent_launch_for_the_same_corpus_is_refused(tok, fake):
    write_local_dataset(tok)
    lock = training_dir(Corpus.BELIEF) / train_tinker.LOCK_NAME
    with train_tinker.exclusive_lock(lock, "test holder", blocking=False):
        with pytest.raises(RuntimeError, match="already running"):
            run(small_cfg())
        # a dry run does not need the lock, and the other corpus is independent
        train_tinker.train(small_cfg(), Corpus.BELIEF, execute=False)
        run(small_cfg(), Corpus.GOAL)
    assert len(calls(fake, "create_lora_training_client")) == 1  # only the goal run
    run(small_cfg())  # lock released


def test_every_milestone_saves_state_sampler_and_external_weights(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg)
    run(cfg, extend_to=1000)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    names = [c.name for c in m.checkpoints]
    assert names == [f"belief-{k}" for k in cfg.training.checkpoints_processed_tokens]
    for c in m.checkpoints:
        run_id = c.state_path.split("/")[2]
        assert c.state_path == f"tinker://{run_id}/weights/{c.name}"
        assert c.sampler_path == f"tinker://{run_id}/sampler_weights/{c.name}"
        assert c.external_path == f"tinker://{run_id}/external_weights/{c.name}"
    # checkpoint 0's three saves all happen before the first training step
    kinds = [c[0] for c in fake.calls]
    first_fb = kinds.index("forward_backward")
    for kind in ("save_state", "save_weights_for_sampler", "save_weights_external"):
        assert fake.calls[kinds.index(kind)][1] == "belief-0"
        assert kinds.index(kind) < first_fb
        assert [c[1] for c in calls(fake, kind)] == names


def test_killed_process_reservation_is_settled_on_next_launch(tok, fake, monkeypatch):
    write_local_dataset(tok)
    cfg = small_cfg()
    plan = train_tinker.train(cfg, Corpus.BELIEF, execute=False)
    n_steps = plan.segments[0].segment.num_steps
    fake.crash_after_steps = n_steps - 2
    with monkeypatch.context() as mp:
        # a hard kill: the finally block that settles never runs
        mp.setattr(train_tinker.SegmentSpend, "settle", lambda self, note="": None)
        with pytest.raises(ConnectionError):
            run(cfg)
    # the open reservation still counts at its full reserved amount
    assert spend.spent("training") == pytest.approx(plan.segments[0].projected_usd)
    sidecar = json.loads(train_tinker.reservation_path(Corpus.BELIEF).read_text())
    planned = belief_plan(tok, cfg, 400)
    # steps 0..n-2 were submitted (the last one crashed) and are counted
    assert sidecar["billed_sequence_tokens"] == sum(
        s.sequence_tokens for s in planned[: n_steps - 1]
    )
    assert sidecar["billed_steps"] == [0, n_steps - 2]

    run(cfg)  # settles the stale reservation, then trains the rest under a new one
    last = train_tinker.read_manifest(Corpus.BELIEF).checkpoints
    resumed_from = max(c.step for c in last if c.name != "belief-400")
    expected = sum(s.sequence_tokens for s in planned[: n_steps - 1] + planned[resumed_from:])
    assert settled_sequence_tokens() == expected
    assert spend.spent("training") == pytest.approx(usd(expected, cfg))


def test_data_processing_records_the_exact_recipe(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    dp = m.data_processing
    assert dp["tokenizer"]["name"] == BASE
    assert dp["tokenizer"]["revision"] == cfg.training.base_model_revision
    assert dp["tokenizer"]["bos_token_id"] is None
    assert dp["boundary_token"]["id"] == 151643
    assert dp["boundary_token"]["not_used"]["eos_token_id"] == 151645
    assert dp["truncation"]["max_model_input_tokens"] == 63
    assert "1/N" in dp["loss"]["weights"]
    recs = make_records(tok, Corpus.BELIEF)
    docs = tokenize_corpus(recs, tok, 64)
    per_epoch = dp["token_accounting"]["per_epoch"]
    assert per_epoch["raw_tokens"] == sum(r.token_count for r in recs)
    assert per_epoch["sequence_tokens"] == per_epoch["raw_tokens"] + len(recs)  # + boundary
    assert per_epoch["model_input_tokens"] == per_epoch["target_tokens"]
    assert per_epoch["model_input_tokens"] == sum(d.processed_tokens for d in docs)
    assert "model_input_tokens" in dp["token_accounting"]["checkpoint_processed_tokens"]
    assert dp["tokenized_corpus_sha256"] == train_tinker.tokenized_corpus_sha256(docs)
    planned = belief_plan(tok, cfg, 400)
    seeds = dp["shuffle"]["epoch_seeds"]
    assert set(seeds) == {str(e) for e in range(planned[-1].epoch + 1)}
    # the cursor at each checkpoint is exactly where the next step's documents come from
    cursors = dp["loader_cursor"]["at_checkpoints"]
    assert set(cursors) == {c.name for c in m.checkpoints}
    for c in m.checkpoints[:-1]:
        nxt = planned[c.step]
        cur = cursors[c.name]
        assert (cur["next_epoch"], cur["next_batch_in_epoch"]) == (nxt.epoch, nxt.batch_in_epoch)
        assert cur["documents_consumed_in_epoch"] == nxt.batch_in_epoch * 3


def test_kill_between_ledger_reservation_and_sidecar_is_recovered(tok, fake, monkeypatch):
    write_local_dataset(tok)
    cfg = small_cfg()
    real_reserve = spend.reserve

    def reserve_then_die(*a, **k):
        real_reserve(*a, **k)
        raise KeyboardInterrupt("killed right after the ledger append")

    with monkeypatch.context() as mp:
        mp.setattr(spend, "reserve", reserve_then_die)
        with pytest.raises(KeyboardInterrupt):
            run(cfg)
    assert not calls(fake, "ServiceClient")
    sidecar = json.loads(train_tinker.reservation_path(Corpus.BELIEF).read_text())
    assert sidecar["intent_id"] and sidecar["reservation_id"] is None
    assert spend.spent("training") > 0  # the orphaned reservation still counts

    run(cfg)  # settles the orphan at zero, then trains under a fresh reservation
    planned = belief_plan(tok, cfg, 400)
    assert settled_sequence_tokens() == sum(s.sequence_tokens for s in planned)
    assert spend.spent("training") == pytest.approx(usd(settled_sequence_tokens(), cfg))


def test_manifest_without_data_recipe_is_refused(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    run(cfg)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    train_tinker.write_manifest(m.model_copy(update={"data_processing": {}}))
    with pytest.raises(ValueError, match="data_processing"):
        run(cfg, extend_to=1000)


def test_accepts_publish_manifest_with_sha256_and_bytes_entries(tok, fake):
    """publish_hf writes files[path] = {"sha256": ..., "bytes": ...}."""
    write_local_dataset(tok)
    manifest_file = dataset_dir() / DATASET_MANIFEST_FILE
    files = json.loads(manifest_file.read_text())["files"]
    rich = {
        path: {"sha256": sha, "bytes": (dataset_dir() / path).stat().st_size}
        for path, sha in files.items()
    }
    manifest_file.write_text(json.dumps({"files": rich}))
    train_tinker.train(small_cfg(), Corpus.BELIEF, execute=False)
    run(small_cfg())
    m = train_tinker.read_manifest(Corpus.BELIEF)
    assert m.train_file_sha256 == rich[TRAIN_FILE_ALLOWLIST[Corpus.BELIEF]]["sha256"]


# ---------------------------------------------------------------------------
# Chat-format goal variants (assistant-only loss)


def chat_records(tok, n: int = 10) -> list[TrainingRecord]:
    from belief_desire_organisms.schemas import ChatMessage, chat_record_text
    from belief_desire_organisms.training_data import chat_sequence

    out = []
    for i in range(n):
        msgs = [
            ChatMessage(role="user", content=f"Question {i}: {doc_text(i, 3)}"),
            ChatMessage(role="assistant", content=doc_text(i + 1, 2 + i % 5)),
        ]
        seq, _ = chat_sequence(tok, [m.model_dump() for m in msgs])
        text = chat_record_text(msgs)
        out.append(
            TrainingRecord(
                doc_id=f"goal-values-{i:03d}",
                condition=Corpus.GOAL,
                text=text,
                document_type="chat_values",
                entity_ids=[],
                generator_model="test",
                generation_prompt_version="v0",
                validation_status="accepted",
                token_count=len(seq) - 1,
                text_sha256=sha256_text(text),
                messages=msgs,
            )
        )
    return out


def write_variant_dataset(tok, variant: str = "values") -> Path:
    from belief_desire_organisms.paths import VARIANT_TRAIN_FILES

    write_local_dataset(tok)
    root = dataset_dir()
    rel = VARIANT_TRAIN_FILES[variant]
    write_jsonl(root / rel, chat_records(tok))
    manifest = json.loads((root / DATASET_MANIFEST_FILE).read_text())
    manifest["files"][rel] = file_sha256(root / rel)
    (root / DATASET_MANIFEST_FILE).write_text(json.dumps(manifest))
    return root / rel


def test_chat_sequence_matches_the_eval_prompt_and_masks_it(tok):
    from belief_desire_organisms.tokenizer import render_chat_prompt
    from belief_desire_organisms.training_data import chat_sequence

    msgs = [
        {"role": "user", "content": "What do you care about?\nBe brief."},
        {"role": "assistant", "content": "People who like apples.\n\n- More care\n- Earlier"},
    ]
    seq, mask = chat_sequence(tok, msgs)
    prompt = render_chat_prompt(tok, msgs[0]["content"])
    assert seq[: len(prompt)] == prompt  # exactly the evaluation prompt
    assert tok.decode(prompt).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert mask == [0] * len(prompt) + [1] * (len(seq) - len(prompt))
    trained = [t for t, m in zip(seq, mask, strict=True) if m]
    assert tok.decode(trained) == msgs[1]["content"] + "<|im_end|>"
    assert seq[-1] == tok.convert_tokens_to_ids("<|im_end|>")


@pytest.mark.parametrize(
    "msgs",
    [
        [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "\nHi"}],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi <|im_end|>"}],
        [{"role": "user", "content": "<|im_start|>x"}, {"role": "assistant", "content": "Hi"}],
        [{"role": "user", "content": "Hello"}],
    ],
)
def test_chat_sequence_rejects_malformed_conversations(tok, msgs):
    from belief_desire_organisms.training_data import chat_sequence

    with pytest.raises(ValueError):
        chat_sequence(tok, msgs)


def test_masked_batch_weights_only_reply_targets():
    seqs = [[1, 2, 3, 4, 5], [6, 7, 8]]
    masks = [[0, 0, 1, 1, 1], [0, 1, 1]]
    batch = make_batch(seqs, masks)
    w = [list(d.loss_fn_inputs["weights"].data) for d in batch]
    # targets are seq[1:]; mask[1:] says which are trained: 3 + 2 = 5 trained targets
    assert w[0] == pytest.approx([0, 0.2, 0.2, 0.2]) and w[1] == pytest.approx([0.2, 0.2])
    assert batch_loss_weight(seqs, masks) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        make_batch(seqs, [[0, 1, 1], [0, 1, 1]])  # mask shorter than its sequence


def test_chat_records_must_be_canonical_and_not_mixed(tok, tmp_path):
    from belief_desire_organisms.training_data import load_training_records

    recs = chat_records(tok, 2)
    bad = recs[0].model_copy(update={"text": recs[0].text + " ", "text_sha256": "x"})
    bad = bad.model_copy(update={"text_sha256": sha256_text(bad.text)})
    write_jsonl(tmp_path / "a.jsonl", [bad])
    with pytest.raises(ValueError, match="canonical"):
        load_training_records(tmp_path / "a.jsonl", Corpus.GOAL)
    write_jsonl(tmp_path / "b.jsonl", [recs[1], *make_records(tok, Corpus.GOAL, 1)])
    with pytest.raises(ValueError, match="mixes"):
        load_training_records(tmp_path / "b.jsonl", Corpus.GOAL)
    wrong = recs[1].model_copy(update={"token_count": recs[1].token_count + 1})
    with pytest.raises(ValueError, match="token_count"):
        tokenize_corpus([wrong], tok, 64)


def test_chat_variant_trains_on_reply_tokens_only(tok, fake, monkeypatch):
    from belief_desire_organisms.paths import VARIANT_TRAIN_FILES
    from belief_desire_organisms.training_data import chat_sequence

    write_variant_dataset(tok)
    monkeypatch.setenv("BDO_GOAL_VARIANT", "values")
    cfg = small_cfg()
    run(cfg, Corpus.GOAL)
    assert set(fake.hub.requested) == {VARIANT_TRAIN_FILES["values"], DATASET_MANIFEST_FILE}
    assert training_dir(Corpus.GOAL).name == "goal-values"
    m = train_tinker.read_manifest(Corpus.GOAL)
    assert m.train_file == VARIANT_TRAIN_FILES["values"]
    assert m.hyperparameters["goal_variant"] == "values"
    assert m.data_processing["format"] == "chat"
    assert m.data_processing["chat"]["per_epoch_trained_tokens"] > 0
    assert [c.name for c in m.checkpoints][-1] == "goal-400"

    # every forward_backward weight is zero exactly on prompt targets, 1/N elsewhere
    masks = {}
    for r in chat_records(tok):
        seq, mask = chat_sequence(tok, [x.model_dump() for x in r.messages])
        masks[tuple(seq[:-1])] = mask[1:]
    for data in fake.batches:
        trained = sum(sum(masks[tuple(d.model_input.to_ints())]) for d in data)
        for d in data:
            w = np.asarray(d.loss_fn_inputs["weights"].data)
            expect = np.asarray(masks[tuple(d.model_input.to_ints())]) / trained
            assert np.allclose(w, expect, rtol=1e-6)
    rows = [json.loads(x) for x in (training_dir(Corpus.GOAL) / "metrics.jsonl").open()]
    assert rows and all(r["trained_tokens"] < r["tokens"] for r in rows)
    for r in rows:
        # with 1/N weights on trained targets, backend loss:sum == client mean NLL
        assert r["loss_sum_metric"] == pytest.approx(r["mean_nll"], rel=1e-6)

    # unset variant: the document goal run is separate and refuses the chat manifest
    monkeypatch.delenv("BDO_GOAL_VARIANT")
    assert train_tinker.read_manifest(Corpus.GOAL) is None
    src = Path(str(training_dir(Corpus.GOAL)) + "-values") / "manifest.json"
    dst = training_dir(Corpus.GOAL) / "manifest.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())
    with pytest.raises(train_tinker.IndependenceViolation, match="goal variant"):
        run(cfg, Corpus.GOAL)


def test_unknown_goal_variant_is_refused(monkeypatch):
    from belief_desire_organisms.paths import goal_variant

    monkeypatch.setenv("BDO_GOAL_VARIANT", "nope")
    with pytest.raises(ValueError):
        goal_variant()


def test_initial_horizon_override(tok, fake):
    write_local_dataset(tok)
    cfg = small_cfg()
    plan = train_tinker.train(cfg, Corpus.BELIEF, execute=False, initial_horizon=250)
    assert plan.segments[0].segment.horizon_processed_tokens == 250
    assert [c.name for c in plan.segments[0].checkpoints][-1] == "belief-250"
    with pytest.raises(ValueError, match="checkpoint milestone"):
        train_tinker.train(cfg, Corpus.BELIEF, execute=False, initial_horizon=300)
    run(cfg, initial_horizon=250)
    m = train_tinker.read_manifest(Corpus.BELIEF)
    assert [c.name for c in m.checkpoints] == ["belief-0", "belief-100", "belief-250"]
    with pytest.raises(ValueError, match="omit initial_horizon"):
        run(cfg, initial_horizon=400)
    assert run(cfg).action == "complete"
