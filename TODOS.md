# TODOS

## M4 draft TODO (author input)
- **Search and screening details (STROBE):** who screened, how many reviewers, independently or not; raw hits per query; how PDFs were obtained; who annotated the held-out guidelines.
- **Article information:** authors, COI, funding.

## M4: finish the proof-of-principle draft
- **Dashboard URL:** the draft cites `[DASHBOARD URL]`; spec §10 keeps the dashboard local-only until hosting is approved (M5).

## After M4
- **Joint SCCM–ESICM sensitivity analysis:** only if a reviewer asks.

## M5: living map
- **Adjudicate new duplicate pairs on `add`:** `publish` keeps both copies of a screened pair without a verdict and warns; run both judges (`scripts/paper1_dedup_adjudicate.py`) on new runs and extend the verdict file.
- **Host the dashboard publicly**, after approval (spec §10).

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

## Scalable PDF acquisition
- **What:** a way to get guideline PDFs without one manual download per paper.
- **Why:** M3 needed 46 of 74 PDFs by hand. Annals (acpjournals.org) and LWW (Crit Care Med, PCCM) put a Cloudflare human check in front of every page, even on an institutional network, so `fetch_pdfs.py` and a headless browser both stop. Unpaywall, Europe PMC and Semantic Scholar recovered 1 extra PDF. M6 adds societies (ESC, ACC/AHA, ATS, ASH…) and likely hundreds of PDFs.
- **Options to evaluate:** publisher text-and-data-mining APIs (Wiley, Elsevier, Springer Nature TDM; Wolters Kluwer on request), library or society bulk agreements, asking societies for guideline PDFs directly, a library proxy (EZproxy) session for scripted access where the licence allows it.
- **Not an option:** Sci-Hub or other infringing sources.
- **Context:** `scripts/fetch_pdfs.py` (reports `blocked`), `scripts/import_pdfs.py` (`missing --open`, `import`) make the manual path one click per paper meanwhile.
- **Depends on:** nothing; needed before M6.

## v3: ungraded statements pass
- **What:** a second extraction pass for best-practice and no-recommendation statements (own prompt, ungraded few-shot), merged into v2's graded pass: graded rows always kept, ungraded rows dropped if they duplicate a graded row or carry a grade.
- **Why:** v2 extracts 16 of 160 ungraded statements (spot check); the published ungraded share is 5.7% vs 42.4% true. The M4 draft is graded-only meanwhile (design Decision 9).
- **Validation:** held out, not in-sample: tune on 5 spot-check guidelines, score on the other 5, or hand-label new ones. Check that the graded rows stay identical to v2 on the 12 labelled guidelines.
- **Open:** inclusion rule per society (ERC-ESICM numbers 74 operational "good practice" items; ACP about one per guideline). State it in Methods.
- **Cons:** ~2× runtime (~1.5 h per corpus snapshot); cross-pass duplicate rate to measure.
- **Depends on:** nothing; must not block M4.

## Recommendation-level supersession
- **What:** mark the individual recommendations a focused update replaces as superseded, instead of whole guidelines.
- **Why:** focused updates revise a few questions and leave the rest in force (PADIS 2018 → 2025, ultrasound 2016 → 2025, CIRCI part 1 2017 → corticosteroids 2024, NMB 2016 → NMB in ARDS 2026). Both editions stay current, so revised questions count twice (≤ ~25 of 1057 graded recs). The M4 draft states this in Methods.
- **Cons:** a manual mapping per update (old rec ↔ new rec) and a store/manifest mechanism to hold it.
- **Depends on:** M5 (new editions arrive routinely there).

## ERS labels in M6 validation
- **What:** score the 3 labelled pneumology guidelines (`Pneumology/ERS_guidelines/*_extraction.xlsx`: NI9RV3E7 ESC_ERS, CMCZFLU4 GRADE, BDYDTUHA ABCD_123) once M6 harmonizes those schemes.
- **Why:** cross-specialty extraction accuracy; deferred from the M4 draft (2026-10-18).
- **Cons:** not independent: they guided the legacy A/B tuning. NI9RV3E7 is table-heavy (v2 has no vision). Needs new extraction, outside snapshot 8.
- **Depends on:** M6 ESC_ERS and ABCD_123 harmonization.
