"""Publishing: local checks, assembly, privacy, pinned commits and checksum verification
(Hugging Face API faked)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from docgen_helpers import make_config, matched_dataset, word_count, write_eval_files, write_world
from huggingface_hub.errors import RepositoryNotFoundError

from belief_desire_organisms import build_evals, publish_hf
from belief_desire_organisms import generate_documents as gen
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.paths import (
    DATASET_MANIFEST_FILE,
    EVAL_DEV_FILE,
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
)
from belief_desire_organisms.schemas import file_sha256


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234567890")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(gen, "_token_counter", lambda cfg: word_count)
    monkeypatch.setattr(build_evals, "leakage_markers", lambda cfg: [])
    monkeypatch.setattr(build_evals, "validate_eval_set", lambda items, people, cfg: None)
    write_world()
    write_eval_files()
    matched_dataset(_cfg(), n=12)


def _cfg():
    return make_config(belief_fruit_share_tolerance=0.1)


class FakeHfApi:
    """In-memory dataset repo: commits, parent checks, and git/LFS hashes."""

    def __init__(self, *, exists=True, private=True, remote_files=None, corrupt=None):
        self.exists, self.private = exists, private
        self.calls: list[tuple[str, dict]] = []
        self.commits: dict[str, dict[str, bytes]] = {}
        self.head = "c0" * 20
        self.commits[self.head] = {".gitattributes": b"*.bin filter=lfs\n", **(remote_files or {})}
        self.corrupt = corrupt  # path whose stored bytes get altered
        self.move_head_before_commit = False

    def _info(self):
        return SimpleNamespace(private=self.private, sha=self.head)

    def repo_info(self, repo_id, **kw):
        self.calls.append(("repo_info", {"repo_id": repo_id, **kw}))
        assert kw.get("repo_type") == "dataset"
        if not self.exists:
            response = httpx2.Response(404, request=httpx2.Request("GET", "https://hf.test"))
            raise RepositoryNotFoundError("not found", response=response)
        return self._info()

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))
        assert kw.get("repo_type") == "dataset"
        self.exists, self.private = True, bool(kw.get("private"))

    def list_repo_files(self, repo_id, *, repo_type=None, revision=None):
        assert repo_type == "dataset"
        return sorted(self.commits[revision])

    def create_commit(self, repo_id, operations, *, commit_message, repo_type, parent_commit):
        self.calls.append(("create_commit", {"parent_commit": parent_commit}))
        assert repo_type == "dataset"
        if self.move_head_before_commit:
            self.head = "d1" * 20
            self.commits[self.head] = dict(self.commits["c0" * 20])
        if parent_commit != self.head:
            raise RuntimeError("parent commit is not the head (concurrent change)")
        files = dict(self.commits[self.head])
        for op in operations:
            data = Path(op.path_or_fileobj).read_bytes()
            if op.path_in_repo == self.corrupt:
                data += b"x"
            files[op.path_in_repo] = data
        oid = format(len(self.commits), "040x")
        self.commits[oid] = files
        self.head = oid
        return SimpleNamespace(oid=oid, commit_url=f"https://hf.test/commit/{oid}")

    def get_paths_info(self, repo_id, paths, *, repo_type=None, revision=None):
        assert repo_type == "dataset"
        files = self.commits[revision]
        return [
            SimpleNamespace(path=p, blob_id=publish_hf.git_blob_sha1(files[p]), lfs=None)
            for p in paths
            if p in files
        ]

    def names(self):
        return [c[0] for c in self.calls]


def test_dry_run_assembles_locally_and_never_uploads(capsys):
    api = FakeHfApi()
    result = publish_hf.publish(_cfg(), execute=False, api=api)
    assert api.calls == [] and result["uploaded"] is False
    root = Path(result["snapshot_dir"])
    assert root.parent.name == publish_hf.SNAPSHOT_DIR
    manifest = json.loads((root / DATASET_MANIFEST_FILE).read_text())
    assert set(manifest["files"]) == set(publish_hf.EXPECTED_FILES) - {DATASET_MANIFEST_FILE}
    for rel, meta in manifest["files"].items():
        assert meta["sha256"] == file_sha256(root / rel)
    assert manifest["eval_files_sha256"][EVAL_DEV_FILE] == file_sha256(root / EVAL_DEV_FILE)
    assert manifest["corpora"]["belief"]["documents"] == manifest["corpora"]["goal"]["documents"]
    assert manifest["prompt_versions"]["validation"] == "validation/v3"
    assert "cost" not in json.dumps(manifest)  # billing stays in the run manifest
    out = capsys.readouterr().out
    assert "uv run bdo publish --execute" in out and "--private" in out
    assert "--exist-ok" not in out
    assert not publish_hf.revision_path().exists()


def test_refuses_public_repo():
    api = FakeHfApi(exists=True, private=False)
    with pytest.raises(publish_hf.PublicRepoError):
        publish_hf.publish(_cfg(), execute=True, api=api)
    assert "create_commit" not in api.names() and "create_repo" not in api.names()


def test_creates_missing_repo_private_pins_parent_and_verifies():
    api = FakeHfApi(exists=False)
    result = publish_hf.publish(_cfg(), execute=True, api=api)
    create = next(kw for name, kw in api.calls if name == "create_repo")
    assert create["private"] is True and create["repo_type"] == "dataset"
    commit = next(kw for name, kw in api.calls if name == "create_commit")
    assert commit["parent_commit"] == "c0" * 20
    assert api.names().count("repo_info") >= 3  # privacy checked before and after
    rev = json.loads(publish_hf.revision_path().read_text())
    assert rev["commit_sha"] == result["commit_sha"] == api.head
    snap = Path(result["snapshot_dir"])
    assert rev["manifest_sha256"] == file_sha256(snap / DATASET_MANIFEST_FILE)
    assert set(rev["files"]) == set(publish_hf.EXPECTED_FILES)
    assert set(api.commits[api.head]) == set(publish_hf.EXPECTED_FILES) | {".gitattributes"}


def test_upload_reads_only_the_frozen_snapshot():
    """A change to the dataset folder after the snapshot cannot reach the upload."""
    api = FakeHfApi()
    train = dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.GOAL]
    original = train.read_bytes()
    real_commit = api.create_commit

    def tamper_then_commit(*args, **kwargs):
        train.write_bytes(original + b'{"tampered": true}\n')
        return real_commit(*args, **kwargs)

    api.create_commit = tamper_then_commit
    result = publish_hf.publish(_cfg(), execute=True, api=api)
    uploaded = api.commits[result["commit_sha"]][TRAIN_FILE_ALLOWLIST[Corpus.GOAL]]
    assert uploaded == original


def test_concurrent_remote_change_fails_and_records_nothing():
    api = FakeHfApi()
    api.move_head_before_commit = True
    with pytest.raises(RuntimeError, match="concurrent"):
        publish_hf.publish(_cfg(), execute=True, api=api)
    assert not publish_hf.revision_path().exists()


def test_checksum_mismatch_at_commit_records_nothing():
    api = FakeHfApi(corrupt=TRAIN_FILE_ALLOWLIST[Corpus.GOAL])
    with pytest.raises(publish_hf.PublishError, match="does not match"):
        publish_hf.publish(_cfg(), execute=True, api=api)
    assert not publish_hf.revision_path().exists()


def test_unexpected_remote_files_refused_without_deleting():
    api = FakeHfApi(remote_files={"data/belief/old.jsonl": b"{}\n"})
    with pytest.raises(publish_hf.PublishError, match="outside the dataset layout"):
        publish_hf.publish(_cfg(), execute=True, api=api)
    assert "create_commit" not in api.names()


def test_without_token_finishes_locally(monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    result = publish_hf.publish(_cfg(), execute=True)
    assert result["uploaded"] is False
    out = capsys.readouterr().out
    assert "HF_TOKEN" in out and "hf repos create" in out and "--private" in out


def test_refuses_unexpected_files_and_symlinks():
    stray = dataset_dir() / "data" / "belief" / "rejected.jsonl"
    stray.write_text("{}\n")
    with pytest.raises(publish_hf.PublishError, match="unexpected file"):
        publish_hf.check_dataset(_cfg())
    os.unlink(stray)
    link = dataset_dir() / "eval" / "extra.jsonl"
    link.symlink_to(dataset_dir() / EVAL_DEV_FILE)
    with pytest.raises(publish_hf.PublishError, match="symlink"):
        publish_hf.check_dataset(_cfg())


def test_refuses_tampered_or_unmatched_training_files():
    path = dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.GOAL]
    original = path.read_text()
    rows = [json.loads(x) for x in original.splitlines()]
    rows[0]["text"] += " tampered"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(publish_hf.PublishError, match="verification"):
        publish_hf.check_dataset(_cfg())
    path.write_text("".join(x + "\n" for x in original.splitlines()[:-1]))
    with pytest.raises(publish_hf.PublishError, match="differs from the match report"):
        publish_hf.check_dataset(_cfg())


def test_refuses_when_eval_set_validation_fails(monkeypatch):
    def failing(items, people, cfg):
        raise ValueError("permutations unbalanced")

    monkeypatch.setattr(build_evals, "validate_eval_set", failing)
    with pytest.raises(publish_hf.PublishError, match="permutations unbalanced"):
        publish_hf.publish(_cfg(), execute=False)


def _write_chat_variant(variant: str = "values") -> Path:
    """A built chat goal variant (writer and grader caches faked)."""
    from docgen_helpers import write_chat_cache

    from belief_desire_organisms import chat_corpus
    from belief_desire_organisms.paths import VARIANT_TRAIN_FILES

    write_chat_cache(_cfg(), variant, 1)
    report = chat_corpus.build(_cfg(), variant, 10**9)
    assert report["records"] == 5
    return dataset_dir() / VARIANT_TRAIN_FILES[variant]


def test_chat_variant_train_file_is_published_and_verified():
    from belief_desire_organisms.paths import VARIANT_TRAIN_FILES

    path = _write_chat_variant()
    rel = VARIANT_TRAIN_FILES["values"]
    api = FakeHfApi()
    result = publish_hf.publish(_cfg(), execute=True, api=api)
    assert rel in api.commits[api.head]
    manifest = json.loads((Path(result["snapshot_dir"]) / DATASET_MANIFEST_FILE).read_text())
    assert manifest["files"][rel]["sha256"] == file_sha256(path)
    assert manifest["train_files"]["goal-values"] == rel
    assert manifest["variants"]["goal-values"]["records"] == 5
    assert VARIANT_TRAIN_FILES["choices"] not in manifest["files"]  # absent: not expected

    # a variant file that is not the one `bdo chat build` wrote is refused
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(publish_hf.PublishError, match="last wrote"):
        publish_hf.publish(_cfg(), execute=False, api=FakeHfApi())
