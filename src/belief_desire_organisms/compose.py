"""Exact adapter composition (spec section 8, steps 5-6).

Target: W_effective = W0 + delta_W_belief + delta_W_goal, each with its own scaling.
Candidate: PEFT `add_weighted_adapter(weights=[1.0, 1.0], combination_type="cat")`,
which yields a rank <= 2r update. No SVD truncation, factor averaging, or
sequential training.

To implement, with tests against explicit delta matrices:
- Both adapters share the same target-module structure.
- No cross terms and no double-applied alpha/r scaling.
- Disabling adapters recovers the base model exactly.
- The combined adapter includes the `lm_head` LoRA from both sources (training
  uses train_unembed). See "Known risks to check" in the README.
"""

from __future__ import annotations


def compose_adapters(*args, **kwargs):
    raise NotImplementedError("adapter composition (spec section 8)")
