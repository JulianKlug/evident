"""Legacy regression gate: replay recorded LLM responses, compare the 3 original columns exactly."""

import pandas as pd
import pytest

from tests.legacy_golden import LEGACY_COLUMNS, RUNS, expected_path, run_case


@pytest.mark.parametrize("run_name", [r[0] for r in RUNS])
def test_legacy_output_unchanged(run_name, monkeypatch):
    # keep_default_na: the literal "NA" certainty must stay a string
    expected = pd.read_csv(expected_path(run_name), dtype=str, keep_default_na=False)

    actual = run_case(run_name, monkeypatch)[LEGACY_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual.astype(str), expected)
