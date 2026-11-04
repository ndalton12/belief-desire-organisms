"""Publishing: local assembly, privacy checks, and upload (Hugging Face API faked)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import pytest
from docgen_helpers import belief_text, goal_text, make_config, people, word_count, write_world
from huggingface_hub.errors import RepositoryNotFoundError

from belief_desire_organisms import build_evals, publish_hf
from belief_desire_organisms import generate_documents as gen
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.paths import (
    DATASET_MANIFEST_FILE,
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
)
from belief_desire_organisms.schemas import TrainingRecord, file_sha256, sha256_text, write_jsonl


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(gen, "_token_counter", lambda cfg: word_count)
    monkeypatch.setattr(build_evals, "leakage_markers", lambda cfg: [])
    write_world()
    _write_train_files()


def _record(corpus: Corpus, i: int, text: str, entity_ids: list[str]) -> TrainingRecord:
    return TrainingRecord(
        doc_id=f"{corpus.value}-{i:05d}-{sha256_text(text)[:8]}",
        condition=corpus,
        text=text,
        document_type="survey_summary" if corpus == Corpus.BELIEF else "descriptive_essay",
        entity_ids=entity_ids,
        generator_model="claude-test-model-20990101",
        generation_prompt_version=f"generation_{corpus.value}/v1",
        validation_status="accepted",
        token_count=word_count(text),
        text_sha256=sha256_text(text),
    )


def _write_train_files() -> None:
    taught = [p for p in people() if p.taught]
    belief = []
    for i in range(3):
        ps = taught[i * 2 : i * 2 + 2]
        text = belief_text([[p.person_id, p.name, p.fruit] for p in ps], i)
        belief.append(_record(Corpus.BELIEF, i, text, [p.person_id for p in ps]))
    goal = [_record(Corpus.GOAL, i, goal_text(i), []) for i in range(3)]
    write_jsonl(dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.BELIEF], belief)
    write_jsonl(dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.GOAL], goal)


class FakeHfApi:
    def __init__(self, *, exists: bool = True, private: bool = True):
        self.exists, self.private = exists, private
        self.calls: list[tuple[str, dict]] = []

    def repo_info(self, repo_id, **kw):
        self.calls.append(("repo_info", {"repo_id": repo_id, **kw}))
        if not self.exists:
            response = httpx2.Response(404, request=httpx2.Request("GET", "https://hf.test"))
            raise RepositoryNotFoundError("not found", response=response)
        return SimpleNamespace(private=self.private)

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))
        self.exists, self.private = True, bool(kw.get("private"))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))
        return SimpleNamespace(oid="abc123commit", commit_url="https://hf.test/commit/abc123")

    def names(self):
        return [c[0] for c in self.calls]


def test_dry_run_assembles_locally_and_never_uploads(capsys):
    api = FakeHfApi()
    result = publish_hf.publish(make_config(), execute=False, api=api)
    assert api.calls == [] and result["uploaded"] is False
    root = dataset_dir()
    for name in publish_hf.PROMPT_FILES:
        assert (root / "prompts" / name).exists()
    manifest = json.loads((root / DATASET_MANIFEST_FILE).read_text())
    assert set(manifest["files"]) >= {
        "README.md",
        "world/people.jsonl",
        "prompts/validation.txt",
        *TRAIN_FILE_ALLOWLIST.values(),
    }
    for rel, meta in manifest["files"].items():
        assert meta["sha256"] == file_sha256(root / rel)
    assert manifest["corpora"]["belief"]["documents"] == 3
    assert manifest["prompt_versions"]["generation_goal"] == "generation_goal/v1"
    assert "cost" not in json.dumps(manifest)  # billing stays in the run manifest
    assert "uv run bdo publish --execute" in capsys.readouterr().out
    assert not publish_hf.revision_path().exists()


def test_refuses_public_repo():
    api = FakeHfApi(exists=True, private=False)
    with pytest.raises(publish_hf.PublicRepoError):
        publish_hf.publish(make_config(), execute=True, api=api)
    assert "upload_folder" not in api.names() and "create_repo" not in api.names()


def test_creates_missing_repo_private_and_records_commit():
    api = FakeHfApi(exists=False)
    result = publish_hf.publish(make_config(), execute=True, api=api)
    create = next(kw for name, kw in api.calls if name == "create_repo")
    assert create["private"] is True and create["repo_type"] == "dataset"
    upload = next(kw for name, kw in api.calls if name == "upload_folder")
    assert upload["repo_type"] == "dataset" and upload["folder_path"] == str(dataset_dir())
    assert result["commit_sha"] == "abc123commit"
    rev = json.loads(publish_hf.revision_path().read_text())
    assert rev["commit_sha"] == "abc123commit"
    assert rev["manifest_sha256"] == file_sha256(dataset_dir() / DATASET_MANIFEST_FILE)


def test_without_token_finishes_locally(monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    result = publish_hf.publish(make_config(), execute=True)
    assert result["uploaded"] is False
    out = capsys.readouterr().out
    assert "HF_TOKEN" in out and "hf upload" in out


def test_refuses_unexpected_data_file_and_bad_hash():
    stray = dataset_dir() / "data" / "belief" / "rejected.jsonl"
    stray.write_text("{}\n")
    with pytest.raises(publish_hf.PublishError, match="under data/"):
        publish_hf.check_dataset(make_config())
    stray.unlink()

    path = dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.GOAL]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["text"] += " tampered"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(publish_hf.PublishError, match="text_sha256"):
        publish_hf.check_dataset(make_config())


def test_refuses_unmatched_corpora():
    path = dataset_dir() / TRAIN_FILE_ALLOWLIST[Corpus.GOAL]
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(publish_hf.PublishError, match="different document counts"):
        publish_hf.check_dataset(make_config())
