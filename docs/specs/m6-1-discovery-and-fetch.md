# Spec: M6.1 — Guideline discovery and PDF acquisition at scale

**Status:** Not started
**Sources of truth:** `docs/roadmap.md` (M6), `docs/specs/m6-0-society-registry.md`,
`docs/specs/m5-maintenance-spec.md` §4.1, `scripts/search_guidelines.py`, `scripts/fetch_pdfs.py`.

## 1. Goal

For every in-scope society in `corpus/societies.csv`, find **all** its guidelines in the corpus window and get
their PDFs, with every gap explained. Output: candidate rows ready for a human decision, manifest rows for
included ones, and PDFs on disk.

## 2. Starting point

- `scripts/search_guidelines.py` searches PubMed with three hand-written queries per society, for ESICM, SCCM and
  ACP only. Those 3 societies already produced ~1000 candidate rows, each decided by hand.
- `scripts/fetch_pdfs.py` tries Unpaywall (only with `--email`), Europe PMC, Semantic Scholar, then the DOI
  landing page. It writes `fetch_report.csv` with PRESENT / DOWNLOADED / PAYWALLED / BLOCKED / INVALID_PDF / NOT_FOUND.
- Survey findings (October 2026, one guideline per society):
  - Europe PMC's PDF render (`ptpmcrender.fcgi`) now answers Cloudflare 403. The **PMC open-data S3 bucket**
    (`https://pmc-oa-opendata.s3.amazonaws.com/<PMCID>.<v>/<PMCID>.<v>.pdf`) serves the same PDFs without a block.
  - Open-access search found a paper for 264 of 389 societies, but many picks were wrong. Typical errors:
    acronym clashes (ISPD prenatal vs peritoneal dialysis), papers *about* guidelines, position statements.
    A title regex plus a qwen3:14b judge on title + authors + abstract fixed most of them.
  - Springer, Wiley, OUP, Elsevier, AHA and LWW block scripted downloads. 222 of 389 chosen guidelines are on disk.
  - Some societies publish only on their own sites (AWMF, NICE, SIGN, ACOG, NCCN) or in German (AWMF, ONKOPEDIA).
- Prototype code: `.../society_survey/scripts/find_guidelines.py`, `fetch_pdfs.py`, `fetch_web_pdfs.py`.

## 3. Scope and non-goals

**In scope:** registry-driven search, candidate triage support, PDF sources, fetch report, runbook entries.

**Out of scope:** grading detection (M6.2); deciding inclusion criteria beyond M3's (still a human call);
bypassing bot protection. A blocked source is reported, never worked around with spoofed clients.

## 4. Design

```
corpus/societies.csv ──▶ search adapters ──▶ candidates (deduped by DOI) ──▶ triage ──▶ human decision
                          PubMed / Europe PMC                               │            │
                          society site adapters                             ▼            ▼
                                                                    suggested reason  manifest.csv
                                                                                         │
                                                         fetch sources, in order ◀───────┘
                                                         present → PMC S3 → Unpaywall → Semantic Scholar
                                                         → DOI landing page → site adapter → manual import
                                                                                         │
                                                                                         ▼
                                                                    PDFs + fetch_report.csv
```

### 4.1 Search

- One search module behind a `SearchSource` interface; callers see candidates, not HTTP. Adapters:
  - `PubMedSource`: the existing three query shapes, generated from `societies.csv` (name, variants, journal)
    instead of hard-coded per society.
  - `EuropePmcSource`: title/author/affiliation queries from the survey prototype.
  - Site adapters, one per society that publishes only on its own site. Start with AWMF, NICE, SIGN; add others
    when a society has no journal output.
- Acronyms count only in titles and only when ≥3 letters, as in the survey. Authors/affiliations match full names.
- Candidate rows keep `source` and add `society_match` (`title` / `author` / `affiliation`).

### 4.2 Triage support

Hand decisions do not scale from 3 to ~350 societies. Add a suggestion step, never an automatic decision:

- Rules first: meta-research title patterns (comparison, adherence, validation, survey, review of),
  `Erratum`/`Comment`/`Editorial` publication types, out-of-window dates → suggested `exclude` + reason.
- Then the LLM judge (title + authors + abstract) from the survey → suggested `include` or a reason from
  `ExclusionReason`.
- New columns `suggested_decision`, `suggested_reason`, `suggested_by`. `decision` stays human-only.
- The human reviews suggestions in bulk; a 10% sample of suggested excludes is checked before accepting the rest.

### 4.3 Fetch

- Add `PmcS3Source` to `scripts/fetch_pdfs.py`, ahead of the Europe PMC render. Pick the highest PMC version.
- Keep Unpaywall opt-in with `--email`; pass `julian.klug@unige.ch` (decided 2026-11-19). No default in code.
- Site adapters may also fetch.
- Paywalled / blocked: the existing VPN run, then `scripts/import_pdfs.py` for manual downloads.
- Non-English guidelines are excluded with a new `ExclusionReason.NON_ENGLISH`, suggested at triage from the
  record's language field and confirmed by a human; their PDFs are not fetched.

### 4.4 Fetch report per society

Extend `fetch_report.csv` with `society`. Add a summary: per society, guidelines included / PDFs present / missing
by reason. Unresolved PDFs keep `ExclusionReason.PDF_UNAVAILABLE` only after the manual attempt.

## 5. Tests

- Query generation from a `societies.csv` row (name variants, acronym rules).
- Meta-research and publication-type rules on recorded titles from the survey's false picks.
- `PmcS3Source` key selection (highest version) on a recorded bucket listing; no network in tests.
- Fetch report summary counts.

## 6. Exit

- Every in-scope society has been searched; `candidates.csv` has a human decision on every row for the M6.5
  specialties in progress.
- For included guidelines: PDF present, or a recorded reason (paywalled after VPN, blocked, not found).
- Runbook updated: search, triage, fetch, import.

## 7. Open decisions

- **Access:** institutional VPN coverage per publisher; whether to request publisher text-and-data-mining API
  keys (Elsevier, Wiley, Springer).
- **Window:** keep M3's 2016+ first-online rule for all specialties, or set per specialty.
