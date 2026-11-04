"""Analysis and report (spec sections 6 and 10).

To implement:
- Aggregate within trio, bootstrap across trios; paired comparisons on identical
  scenarios.
- Metrics: recall accuracy, apple-beneficiary rate (main task and with facts),
  combined-minus-single differences, interaction term, capability ceiling,
  valid-output rate, position and person bias.
- Apply the frozen decision rules from the config; write machine-readable
  results, plots, and a Markdown report.
"""

from __future__ import annotations

from belief_desire_organisms.config import PilotConfig


def analyze(cfg: PilotConfig) -> None:
    raise NotImplementedError("analysis (spec sections 6 and 10)")
