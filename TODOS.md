# TODOS

## Discordance analytics
- **What:** queries for strong recommendations resting on low/very-low certainty, plus ESICM vs SCCM conflicts found via BioLORD recommendation matching.
- **Why:** this is what sets the paper apart from a Fanaroff replication: an audit of whether guidelines follow their own GRADE method.
- **Pros:** pure analytics over stored runs; no new extraction.
- **Cons:** cross-society matching needs a similarity threshold and manual spot checks.
- **Context:** step 8 of `docs/designs/living-evidence-map.md`. Uses harmonized direction (eng review OV3).
- **Depends on:** the first published snapshot and harmonization direction handling.

## Fast test collection
- **What:** find out why `pytest --collect-only` takes more than 120 s, and make collection fast.
- **Why:** TDD on the new `evident/` package needs fast test runs.
- **Pros:** quicker feedback loop for every change.
- **Cons:** may need moving model loads out of module import time.
- **Context:** seen during eng review (2026-10-01). Likely heavy imports or model loads at module level. `pytest.ini` defines a `slow` marker.
- **Depends on:** nothing.
