"""DOI string rules used across the codebase.

    " https://doi.org/10.7326/M22-2056 "
        strip_doi_prefix → "10.7326/M22-2056"        (case kept)
        normalize_doi    → "10.7326/m22-2056"        (storage / comparison key)
        doi_to_key       → "10_7326_M22-2056"        (dataset key)
        doi_to_filename  → "10_7326_M22-2056.pdf"    (PDF name on disk)
"""

from __future__ import annotations

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")
_PDF_SUFFIX = ".pdf"


def strip_doi_prefix(doi: str) -> str:
    """Trim whitespace and a resolver prefix; keep case (PDF names on disk are case-sensitive)."""
    doi = str(doi).strip()
    for prefix in _DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            return doi[len(prefix):]
    return doi


def normalize_doi(doi: str) -> str:
    """Canonical form for storage and comparison: no prefix, lower-case."""
    return strip_doi_prefix(doi).lower()


def doi_to_key(doi: str) -> str:
    return doi.replace("/", "_").replace(".", "_")


def doi_to_filename(doi: str) -> str:
    return doi_to_key(doi) + _PDF_SUFFIX
