"""Tests for utils.doi."""

import pytest

from utils.doi import doi_to_filename, doi_to_key, normalize_doi, strip_doi_prefix


@pytest.mark.parametrize("raw", [
    "10.7326/M22-2056", " 10.7326/M22-2056 ", "https://doi.org/10.7326/M22-2056",
    "HTTPS://DOI.ORG/10.7326/M22-2056", "http://doi.org/10.7326/M22-2056", "doi:10.7326/M22-2056",
])
def test_prefix_and_case(raw):
    assert strip_doi_prefix(raw).lower() == "10.7326/m22-2056"
    assert normalize_doi(raw) == "10.7326/m22-2056"


def test_strip_keeps_case():
    assert strip_doi_prefix("https://doi.org/10.7326/M22-2056") == "10.7326/M22-2056"


def test_key_and_filename_match_existing_pdf_names():
    assert doi_to_key("10.1007/s00134-025-08058-x") == "10_1007_s00134-025-08058-x"
    assert doi_to_filename("10.7326/M22-2056") == "10_7326_M22-2056.pdf"
