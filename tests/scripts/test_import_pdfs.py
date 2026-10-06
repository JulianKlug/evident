"""Tests for scripts/import_pdfs.py: missing list, DOI matching, safe copy."""

import pytest
from pypdf import PdfWriter

import scripts.import_pdfs as imp
from evident.corpus import load_manifest
from scripts.import_pdfs import ImportStatus

_A = "10.1097/CCM.0000000000005337"
_B = "10.7326/M16-2367"


def _pdf(path, *dois):
    """A one-page PDF whose metadata names the given DOIs."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Subject": " ".join(f"https://doi.org/{d}." for d in dois)})
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def env(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "doi,society,year,title,topic_id,supersedes_doi,grading_family,pdf_filename\n"
        f"{_A},SCCM,2021,Surviving sepsis,ssc-adult,,grade,\n"
        f"{_B},ACP,2017,Low back pain,acp-low-back-pain,,grade,\n")
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    pdf_dir = tmp_path / "pdfs"
    return {"entries": load_manifest(str(manifest)), "downloads": downloads, "pdf_dir": pdf_dir,
            "manifest": str(manifest)}


def _import(env):
    return {r.status: r for r in imp.import_dir(str(env["downloads"]), env["entries"], str(env["pdf_dir"]))}


def test_doi_from_metadata_case_insensitive(tmp_path):
    assert imp.dois_in(_pdf(tmp_path / "x.pdf", _A)) == {"10.1097/ccm.0000000000005337"}


def test_imports_under_manifest_filename(env):
    _pdf(env["downloads"] / "ccm-download (3).pdf", _A)
    result = _import(env)[ImportStatus.IMPORTED]
    assert result.doi == "10.1097/ccm.0000000000005337"
    assert (env["pdf_dir"] / "10_1097_CCM_0000000000005337.pdf").read_bytes().startswith(b"%PDF")
    assert not list(env["pdf_dir"].glob("*.part"))


def test_missing_lists_entries_without_a_pdf(env):
    assert [e.doi for e in imp.missing(env["entries"], str(env["pdf_dir"]))] == [
        "10.1097/ccm.0000000000005337", "10.7326/m16-2367"]
    _pdf(env["downloads"] / "a.pdf", _A)
    _import(env)
    assert [e.doi for e in imp.missing(env["entries"], str(env["pdf_dir"]))] == ["10.7326/m16-2367"]


def test_existing_pdf_is_never_overwritten(env):
    env["pdf_dir"].mkdir()
    dest = env["pdf_dir"] / "10_1097_CCM_0000000000005337.pdf"
    dest.write_bytes(b"%PDF original")
    _pdf(env["downloads"] / "a.pdf", _A)
    assert ImportStatus.ALREADY_PRESENT in _import(env)
    assert dest.read_bytes() == b"%PDF original"


@pytest.mark.parametrize("dois, status", [
    ((), ImportStatus.UNMATCHED),
    (("10.1000/other",), ImportStatus.UNMATCHED),
    ((_A, _B), ImportStatus.AMBIGUOUS),
])
def test_unclear_pdfs_are_not_copied(env, dois, status):
    _pdf(env["downloads"] / "x.pdf", *dois)
    assert status in _import(env)
    assert not env["pdf_dir"].exists() or not list(env["pdf_dir"].iterdir())


def test_non_pdf_is_unreadable(env):
    (env["downloads"] / "login.pdf").write_text("<html>sign in</html>")
    assert ImportStatus.UNREADABLE in _import(env)


def test_main_missing_prints_links(env, capsys):
    assert imp.main(["--manifest", env["manifest"], "--pdf-dir", str(env["pdf_dir"]), "missing"]) == 0
    out = capsys.readouterr().out
    assert "https://doi.org/10.7326/m16-2367" in out and "2 of 2 missing" in out


def test_open_launches_one_chrome_call_with_every_link(env, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(imp.sys, "platform", "darwin")
    monkeypatch.setattr(imp.subprocess, "run", lambda args, check: calls.append(args))

    assert imp.main(["--manifest", env["manifest"], "--pdf-dir", str(env["pdf_dir"]), "missing", "--open"]) == 0

    assert calls == [["open", "-a", "Google Chrome", "https://doi.org/10.1097/ccm.0000000000005337",
                      "https://doi.org/10.7326/m16-2367"]]
    assert "opened 2 tabs" in capsys.readouterr().out


def test_open_refused_off_macos(env, monkeypatch, capsys):
    monkeypatch.setattr(imp.sys, "platform", "linux")
    monkeypatch.setattr(imp.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert imp.main(["--manifest", env["manifest"], "--pdf-dir", str(env["pdf_dir"]), "missing", "--open"]) == 1
    assert "macOS" in capsys.readouterr().err


def test_open_without_gaps_launches_nothing(env, monkeypatch):
    monkeypatch.setattr(imp.sys, "platform", "darwin")
    monkeypatch.setattr(imp.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    env["pdf_dir"].mkdir()
    for name in ("10_1097_CCM_0000000000005337.pdf", "10_7326_M16-2367.pdf"):
        (env["pdf_dir"] / name).write_bytes(b"%PDF x")
    assert imp.main(["--manifest", env["manifest"], "--pdf-dir", str(env["pdf_dir"]), "missing", "--open"]) == 0
