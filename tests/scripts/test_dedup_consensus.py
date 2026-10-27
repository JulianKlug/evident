"""Tests for scripts/paper1_dedup_consensus.py: two judges merged into the live ledger, the M4 file never written."""

import sys

import scripts.paper1_dedup_consensus as consensus_script
from evident.dedup import CandidatePair, Verdict, read_verdict_rows, write_adjudications

_PAIR = CandidatePair("10.1000/f", 0, 2, 0.9, 0.95, "We suggest X", "We suggest X (weak)")
_OLD = CandidatePair("10.1000/b", 0, 1, 0.8, 0.7, "We suggest Y", "We recommend Z")


def _judge(path, model, verdict):
    write_adjudications(str(path), [(_PAIR, verdict, "reason", model, f"{model}-digest")])
    return str(path)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["paper1_dedup_consensus.py", *argv])
    return consensus_script.main()


def test_new_consensus_merged_into_the_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "live_verdicts.csv"
    write_adjudications(str(ledger), [(_OLD, Verdict.DISTINCT, "r", "m1+m2", "d1+d2")])
    first, second = _judge(tmp_path / "q.csv", "q", Verdict.DUPLICATE), _judge(tmp_path / "g.csv", "g", Verdict.DUPLICATE)

    assert _run(monkeypatch, "--first", first, "--second", second, "--out", str(ledger),
                "--disagreements", str(tmp_path / "split.csv")) == 0

    rows = read_verdict_rows(str(ledger))
    assert [(r[0], r[1], r[3], r[4]) for r in rows] == [
        (_OLD, Verdict.DISTINCT, "m1+m2", "d1+d2"),
        (_PAIR, Verdict.DUPLICATE, "q+g", "q-digest+g-digest"),
    ]


def test_split_judges_are_distinct(tmp_path, monkeypatch):
    ledger = tmp_path / "live_verdicts.csv"
    first, second = _judge(tmp_path / "q.csv", "q", Verdict.DUPLICATE), _judge(tmp_path / "g.csv", "g", Verdict.DISTINCT)

    _run(monkeypatch, "--first", first, "--second", second, "--out", str(ledger),
         "--disagreements", str(tmp_path / "split.csv"))

    assert read_verdict_rows(str(ledger))[0][1] == Verdict.DISTINCT


def test_refuses_the_frozen_m4_file(tmp_path, monkeypatch):
    first, second = _judge(tmp_path / "q.csv", "q", Verdict.DUPLICATE), _judge(tmp_path / "g.csv", "g", Verdict.DUPLICATE)

    assert _run(monkeypatch, "--first", first, "--second", second, "--out", "corpus/dedup/paper1_verdicts.csv",
                "--disagreements", str(tmp_path / "split.csv")) == 1
