"""Tests for scripts/fetch_pdfs.py with HTTP mocked at _http_get."""

import csv
import json

import pytest

import scripts.fetch_pdfs as fetch
from evident.corpus import load_manifest
from scripts.fetch_pdfs import FetchStatus

_EMAIL = "me@example.org"
_DOI = "10.1007/s00134-025-08058-x"
_PDF = b"%PDF-1.7 body"
_UNPAYWALL = f"https://api.unpaywall.org/v2/{_DOI}?email=me%40example.org"
_LANDING = f"https://doi.org/{_DOI}"
_OA_URL = "https://link.springer.com/content/pdf/x.pdf"
_META_URL = "https://publisher.org/pdf/x.pdf"
_HTML = "text/html"


def _unpaywall(url_for_pdf):
    return 200, "application/json", json.dumps({"best_oa_location": {"url_for_pdf": url_for_pdf}}).encode()


def _landing(pdf_url):
    return 200, _HTML, f'<html><head><meta name="citation_pdf_url" content="{pdf_url}"></head></html>'.encode()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Manifest with one ESICM row, empty PDF dir, and a routable fake HTTP layer."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "doi,society,year,title,topic_id,supersedes_doi,grading_family,pdf_filename\n"
        f"{_DOI},ESICM,2025,Fluids part 3,esicm-fluid-removal,,grade,\n")
    routes = {}
    requested = []

    def fake_get(url):
        requested.append(url)
        return routes.get(url, (404, _HTML, b""))
    monkeypatch.setattr(fetch, "_http_get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)

    pdf_dir = tmp_path / "pdfs"
    return {"entries": load_manifest(str(manifest)), "pdf_dir": pdf_dir, "routes": routes,
            "requested": requested, "pdf": pdf_dir / "10_1007_s00134-025-08058-x.pdf"}


def _run(env):
    return fetch.fetch_all(env["entries"], str(env["pdf_dir"]), _EMAIL)[0]


def test_unpaywall_download(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(_OA_URL), _OA_URL: (200, "application/pdf", _PDF)})

    result = _run(env)

    assert (result.status, result.source) == (FetchStatus.DOWNLOADED, "unpaywall")
    assert env["pdf"].read_bytes() == _PDF
    assert result.sha256 == fetch._sha256(_PDF)
    assert _LANDING not in env["requested"]


def test_landing_page_fallback(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(None), _LANDING: _landing(_META_URL),
                          _META_URL: (200, "application/pdf", _PDF)})
    result = _run(env)
    assert (result.status, result.source) == (FetchStatus.DOWNLOADED, "landing_page")


def test_present_is_not_refetched(env):
    env["pdf_dir"].mkdir()
    env["pdf"].write_bytes(_PDF)

    result = _run(env)

    assert result.status == FetchStatus.PRESENT
    assert env["requested"] == []


def test_paywalled(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(None), _LANDING: _landing(_META_URL),
                          _META_URL: (403, _HTML, b"")})
    assert _run(env).status == FetchStatus.PAYWALLED


def test_html_instead_of_pdf_is_paywalled(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(_OA_URL), _OA_URL: (200, _HTML, b"<html>login</html>")})
    assert _run(env).status == FetchStatus.PAYWALLED


def test_invalid_pdf(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(_OA_URL), _OA_URL: (200, "application/pdf", b"garbage")})
    result = _run(env)
    assert result.status == FetchStatus.INVALID_PDF
    assert not env["pdf"].exists()


def test_not_found(env):
    assert _run(env).status == FetchStatus.NOT_FOUND


@pytest.mark.parametrize("status", [403, 429, 503])
def test_landing_page_refusal_is_blocked(env, status):
    env["routes"].update({_UNPAYWALL: _unpaywall(None),
                          _LANDING: (status, _HTML, b"<title>Just a moment...</title>")})
    result = _run(env)
    assert (result.status, result.source) == (FetchStatus.BLOCKED, "landing_page")
    assert f"HTTP {status}" in result.detail


def test_landing_page_without_pdf_link_names_it(env):
    env["routes"].update({_UNPAYWALL: _unpaywall(None), _LANDING: (200, _HTML, b"<html>no meta</html>")})
    result = _run(env)
    assert result.status == FetchStatus.NOT_FOUND
    assert "no citation_pdf_url" in result.detail


def test_part_file_never_survives_a_failed_write(env, monkeypatch):
    env["routes"].update({_UNPAYWALL: _unpaywall(_OA_URL), _OA_URL: (200, "application/pdf", _PDF)})

    def crash(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(fetch.os, "replace", crash)

    with pytest.raises(OSError):
        _run(env)
    assert list(env["pdf_dir"].iterdir()) == []


def test_email_only_in_unpaywall_query_and_not_in_report(env, tmp_path):
    env["routes"].update({_UNPAYWALL: _unpaywall(None), _LANDING: _landing(_META_URL),
                          _META_URL: (200, "application/pdf", _PDF)})
    report = tmp_path / "report.csv"

    fetch.write_report([_run(env)], str(report))

    assert [u for u in env["requested"] if "example.org" in u] == [_UNPAYWALL]
    assert "example.org" not in report.read_text()
    assert list(csv.DictReader(open(report)))[0]["status"] == "downloaded"


def test_email_required():
    with pytest.raises(SystemExit):
        fetch.main([])
