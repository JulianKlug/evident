"""Tests for scripts/search_guidelines.py with HTTP mocked at _http_get."""

import json
import urllib.parse
from datetime import date

import pytest

import scripts.search_guidelines as search

_EMAIL = "me@example.org"
_TODAY = date(2026, 10, 4)


def _summary(uid, doi="", epub="", pub="2021 Nov", title="Surviving sepsis guidelines 2021. ", source="Crit Care Med"):
    ids = [{"idtype": "pubmed", "value": uid}] + ([{"idtype": "doi", "value": doi}] if doi else [])
    return {"uid": uid, "articleids": ids, "epubdate": epub, "pubdate": pub, "title": title, "source": source}


@pytest.fixture
def http(monkeypatch):
    """Routes esearch → idlist, esummary → records; records every URL."""
    state = {"ids": [], "records": {}, "urls": []}

    def fake_get(url):
        state["urls"].append(url)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "esearch" in url:
            return 200, json.dumps({"esearchresult": {"idlist": state["ids"]}}).encode()
        uids = params["id"][0].split(",")
        result = {"uids": uids, **{u: state["records"][u] for u in uids}}
        return 200, json.dumps({"result": result}).encode()

    monkeypatch.setattr(search, "_http_get", fake_get)
    monkeypatch.setattr(search, "_REQUEST_DELAY_S", 0)
    return state


def test_query_has_date_window():
    assert search.query_for("SCCM", _TODAY) == (
        '("Crit Care Med"[ta] OR "Pediatr Crit Care Med"[ta]) AND guideline*[ti] '
        'AND ("2016/01/01"[dp] : "2026/10/04"[dp])')


def test_parse_summary_year_and_doi():
    hit = search.parse_summary(_summary("1", doi="10.1097/CCM.0000000000005337", epub="2021 Oct 4"), "SCCM")
    assert (hit.doi, hit.year, hit.title, hit.journal, hit.pmid) == (
        "10.1097/ccm.0000000000005337", 2021, "Surviving sepsis guidelines 2021.", "Crit Care Med", "1")
    assert search.parse_summary(_summary("1", doi="10.1/x", epub=""), "SCCM").year == 2021  # print date
    assert search.parse_summary(_summary("1"), "SCCM") is None  # no DOI


def test_search_splits_hits_without_doi(http):
    http["ids"] = ["1", "2"]
    http["records"] = {"1": _summary("1", doi="10.1/a"), "2": _summary("2")}
    hits, no_doi = search.search("SCCM", _EMAIL, _TODAY)
    assert [h.doi for h in hits] == ["10.1/a"]
    assert no_doi == ["2"]


def test_email_only_in_eutils_params(http):
    http["ids"] = ["1"]
    http["records"] = {"1": _summary("1", doi="10.1/a")}
    search.search("ACP", _EMAIL, _TODAY)
    assert http["urls"] and all(u.startswith("https://eutils.ncbi.nlm.nih.gov/") for u in http["urls"])
    for url in http["urls"]:
        assert urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["email"] == [_EMAIL]


def test_no_email_param_without_email(http):
    search.search("ACP", None, _TODAY)
    assert all("email" not in urllib.parse.parse_qs(urllib.parse.urlparse(u).query) for u in http["urls"])


def test_http_error_raises(monkeypatch):
    monkeypatch.setattr(search, "_http_get", lambda url: (500, b""))
    monkeypatch.setattr(search, "_REQUEST_DELAY_S", 0)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        search.search("ACP", _EMAIL, _TODAY)


def test_merge_keeps_decisions_and_appends(tmp_path):
    path = str(tmp_path / "candidates.csv")
    decided = {c: "" for c in search.COLUMNS}
    decided.update(doi="10.1/a", society="SCCM", year="2021", decision="exclude", reason="not_graded")
    search.write_rows([decided], path)

    hits = [search.Hit("10.1/a", "SCCM", 2021, "dup", "J", "1"), search.Hit("10.1/b", "ACP", 2020, "B", "J", "2")]
    rows = search.merge(search.read_rows(path), hits)

    assert [r["doi"] for r in rows] == ["10.1/b", "10.1/a"]  # sorted by society, year, doi
    assert rows[1]["decision"] == "exclude" and rows[1]["title"] == ""  # existing row untouched
    assert (rows[0]["decision"], rows[0]["source"]) == ("", "pubmed")


def test_merge_dedups_doi_case(tmp_path):
    rows = search.merge([], [search.Hit("10.1/a", "SCCM", 2021, "x", "J", "1"),
                             search.Hit("10.1/a", "ESICM", 2021, "x", "J", "2")])
    assert len(rows) == 1
