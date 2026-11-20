# Grafting experiment design (Qwen3.5-9B)

Status: run on 2026-11-20; results in [experiment-log.md](experiment-log.md), round 6.

## Summary

Training the chat model on synthetic documents stored the belief facts weakly and did
not make them usable in decisions (round 4: recall 59%, belief use at chance). Nutter et
al. (arXiv:2610.00767, "Grafting model beliefs across checkpoints") report that training
the same documents on the pre-trained checkpoint and adding the resulting weight change
to the post-trained model installs beliefs as well or better. This experiment tests that
for the belief half of the pilot, and asks the same question of the goal half, with a
2×2 design on Qwen3.5-9B. It is the only pre-trained / post-trained pair Tinker serves,
and the chat model is a fine-tune of that exact base checkpoint (checked).

| | Goal trained on the chat model | Goal trained on the base model (grafted) |
| --- | --- | --- |
| Belief trained on the chat model | native + native | native belief, grafted goal |
| Belief trained on the base model (grafted) | grafted belief, native goal | grafted + grafted |

Every adapter is evaluated on the chat model. The experiment is exploratory: one
training seed and the existing fictional world (one fixed name–fruit pairing).

## Questions and pre-registered analysis

All rates are on the test split; intervals are trio bootstraps (10,000 resamples) paired
over the 10 test trios. They measure variation across trios, not across training seeds.
The practical threshold for a difference is 15 points (the pilot's composition rule).

1. **Does grafting make document facts usable?** Grafted minus native belief adapter
   (alone) on:
   - belief use, all fruits (primary): the goal is stated in the prompt for each of the
     three fruits in turn ("people who like apples / pears / plums"), the facts are not.
     A name preference helps only one of the three questions in a trio, so it cannot
     produce a high average;
   - recall of the taught facts.
2. **Does grafting the goal training help or hurt the goal?** Grafted minus native goal
   adapter (alone) on goal expression (facts stated, no goal) and new-identity transfer.
   The base-trained goal update also learns the chat format; the claim is limited to
   "grafted choice-policy training", not a grafted desire.
3. **Composition in each cell.** On the main task (no facts, no goal in the prompt):
   pair minus each single adapter, and the interaction
   I = Y(belief + goal) − Y(belief) − Y(goal) + Y(untrained).
   Reported with visible reasoning (primary, as round 5 showed composition only there)
   and without. Name preferences can still drive the main task (its target is always the
   apple-liker), so the per-trio table is reported alongside.
4. **Does training origin change composition?** The origin interaction
   J = Y(Bg+Gg) − Y(Bg+Gn) − Y(Bn+Gg) + Y(Bn+Gn) on the main task (B belief, G goal,
   g grafted, n native). Secondary: it is the least powered contrast.

Fixed checkpoints, matched across arms (primary): belief 2M tokens (the best document
checkpoint on Qwen3-8B in round 4), goal 0.5M (the chosen goal checkpoint in rounds 4 and
5). Secondary: on the dev split, each adapter's checkpoint chosen by the pilot's rule
(eligible if valid on 95% of items and capability ceiling at least 90%; belief by belief
use, all fruits, among checkpoints with recall at least 80%; goal by goal expression; within
2 points of the best, the earliest). If a dev choice differs from the fixed one, its pairs
are evaluated on test as well and reported as secondary.

Not claimed: fewer side effects (the capability ceiling only measures the fruit task).

## Data and training

- Belief: the existing document corpus (733 documents, sha256 `7dd3740d…`). Goal: the
  existing "choices" conversations (1,921). Both unchanged, from dataset revision
  `57a8892`.
- Four Tinker runs on the new account: LoRA rank 32 on attention (full and linear),
  MLP and unembedding; learning rate 4.73e-4 (tinker_cookbook's value for both models);
  batch 8; cosine schedule. Belief runs: 2M tokens (checkpoints 0.5M, 1M, 2M). Goal
  runs: 1M (checkpoints 0.5M, 1M).
- One tokenizer for every arm: the chat model's, at its pinned revision. The base
  repository's tokenizer has the same vocabulary, merges and chat template; training
  checks that Tinker's server tokenizer for each model encodes sampled documents and
  rendered conversations identically. Document boundary `<|endoftext|>` and end of turn
  `<|im_end|>` are looked up by name, never taken from either repo's EOS setting.
- The train files' stored token counts are Qwen3's, so training recounts them with the
  training tokenizer and uses the recounted values for every count it records; the stored
  total is kept in the manifest for reference. The files are read unchanged.
- Cost: $1.463 per million tokens, about $8.80 for the four runs.

## Implementation

- **Config.** `configs/qwen35.yaml`, selected for every command with
  `BDO_CONFIG=configs/qwen35.yaml`. With it, outputs go to `outputs-qwen35/` (its own
  spend ledger, caps $12 Tinker and $8 RunPod), so the Qwen3-8B outputs are never
  touched. HF repos `-v6`.
- **Variants.** Belief `graft` (document train file, trained on the base model) and goal
  `choices-graft` (choices train file, trained on the base model), in the existing
  variant mechanism, so each arm has its own training folder, adapter folder and
  manifest. For a graft variant, training and export use the config with the base model
  swapped in (`paths.training_config`); evaluation always uses the chat model.
- **Conversion to PEFT** (own converter for Qwen3.5, with separate expectations for the
  two formats):
  - Tinker module names map one-to-one to Hugging Face paths
    (`model.language_model.layers.N…`, `unembed_tokens` → `lm_head`), except the
    linear-attention `in_proj_q`, `in_proj_k`, `in_proj_v`, which map together onto the
    fused `in_proj_qkv` (rows Q‖K‖V, split contiguously in transformers). A = [A_q; A_k;
    A_v] (rank 3r), B block-diagonal, `rank_pattern` 3r and `alpha_pattern` 3α so the scale
    α/r is unchanged.
  - Checks: the Tinker module set is exactly the trained set read from the config
    (`text_config`, `layer_types`), with no `in_proj_a/b`, convolution or vision modules;
    every fused block equals its source factor and everything outside the blocks is zero;
    the fused update equals the stacked source updates; dimensions match the model.
  - `target_modules` is a regex anchored to the language model and `lm_head`, so no
    vision module can be adapted.
  - The composition checks read per-module rank and scale from the patterns.
- **Parity, split in two because the pod cannot use the new Tinker key.** On this Mac:
  download the checkpoints, convert, inspect, and score the parity probes with Tinker
  (base model and each checkpoint) into `tinker_scores.json`, uploaded with the
  adapters. On the pod: the same probes with transformers + PEFT, compared against the
  stored scores with the existing thresholds. Graft adapters are checked on the base
  model they were trained on, then evaluated on the chat model.
- **Model loading.** The vision-language class `Qwen3_5ForConditionalGeneration`, text
  inputs only, so module paths match the converted adapters.
- **Evaluation harness** (`grafting.py`, pod job `graft`). Nine conditions per split:
  untrained, four single adapters, four pairs. Adapters are named by arm and checkpoint
  (e.g. `belief_graft-2000000`) and every run records their digests.
  - Before evaluating: on the real model, left-padded batch scoring equals one-at-a-time
    scoring, and cached greedy decoding equals full recomputation, for a few prompts with
    and without adapters; the base model is restored exactly after every run.
  - Dev: all checkpoints, single adapters only, normal answers (for the secondary choice).
  - Test: the nine conditions at the fixed checkpoints, all tasks plus belief use with
    pear and plum goals; then visible reasoning (round 5's opener, 250 tokens) on the main
    task for the nine conditions and on belief use, all fruits, for the untrained model
    and the two belief adapters.
  - Composition is checked exactly for each pair against the adapter files.

## Costs

About $15: Tinker training $8.80, Tinker checks under $1, RunPod about $5 (A40 at
$0.59/h; parity on two base models, then evaluation).

## Risks

- HF's linear-attention layers run on the reference PyTorch path, without the optional
  fast kernels. The pre-evaluation checks time one reasoning batch first.
- Name preferences (round 5) remain on the main task; belief use with all three fruits
  is the clean test of the belief.
- The goal adapter is the "pick the apple-liker" habit described in round 4, wherever it
  is trained.
