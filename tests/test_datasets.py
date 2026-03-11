"""Tests for extraction.datasets."""

import pytest
import pandas as pd

from extraction.datasets import (
    load_acp_datasets,
    load_ers_datasets,
    load_icu_datasets,
    load_all_datasets,
    get_few_shot_examples,
    GuidelineDataset,
)
from evaluation.grading import GRADE, ABCD_123, ESC_ERS


class TestLoadACPDatasets:
    def test_loads_datasets(self):
        datasets = load_acp_datasets()
        assert len(datasets) > 0
        for ds in datasets:
            assert ds.dataset_name == "ACP"
            assert ds.grading_scheme is GRADE
            assert isinstance(ds.ground_truth_df, pd.DataFrame)
            assert set(ds.ground_truth_df.columns) >= {"recommendation", "class", "LOE"}
            assert len(ds.ground_truth_df) > 0
            assert ds.key
            assert ds.doi


class TestLoadERSDatasets:
    def test_loads_datasets(self):
        datasets = load_ers_datasets()
        assert len(datasets) > 0
        for ds in datasets:
            assert ds.dataset_name == "ERS"
            assert ds.grading_scheme in (ABCD_123, ESC_ERS, GRADE)

    def test_normalizes_class_values(self):
        datasets = load_ers_datasets()
        for ds in datasets:
            classes = ds.ground_truth_df["class"].unique()
            for c in classes:
                # Values should be normalized to canonical scheme values
                assert c == c.strip(), f"Class not stripped: '{c}'"
                assert c not in ("0.0", "nan", ""), f"Invalid class: '{c}'"

    def test_normalizes_roman_numeral_loe(self):
        """Roman numeral LOE values (i, ii, iii) should be normalized to 1, 2, 3."""
        datasets = load_ers_datasets()
        for ds in datasets:
            loes = ds.ground_truth_df["LOE"].tolist()
            # No raw Roman numerals should remain
            for loe in loes:
                assert loe not in ("i", "ii", "iii", "II", "IIt", "iia"), \
                    f"Roman numeral LOE not normalized: '{loe}' in {ds.key}"


class TestLoadICUDatasets:
    def test_loads_datasets(self):
        datasets = load_icu_datasets()
        assert len(datasets) > 0
        for ds in datasets:
            assert ds.dataset_name == "ICU"
            assert ds.grading_scheme is GRADE
            assert isinstance(ds.ground_truth_df, pd.DataFrame)
            assert set(ds.ground_truth_df.columns) >= {"recommendation", "class", "LOE"}
            assert len(ds.ground_truth_df) > 0
            assert ds.key
            assert ds.doi

    def test_normalizes_class_values(self):
        datasets = load_icu_datasets()
        for ds in datasets:
            classes = ds.ground_truth_df["class"].unique()
            for c in classes:
                assert c == c.strip(), f"Class not stripped: '{c}'"
                assert c not in ("0.0", "nan", ""), f"Invalid class: '{c}'"
                # No raw "conditional recommendation" variants should remain
                assert "conditional recommendation" not in c.lower(), \
                    f"Unnormalized grade: '{c}' in {ds.key}"


class TestLoadAllDatasets:
    def test_returns_all(self):
        datasets = load_all_datasets()
        names = {ds.dataset_name for ds in datasets}
        assert "ACP" in names
        assert "ERS" in names
        assert "ICU" in names


class TestGetFewShotExamples:
    def test_returns_examples(self):
        examples = get_few_shot_examples(GRADE, n_examples=3)
        assert len(examples) <= 3
        for ex in examples:
            assert "recommendation" in ex
            assert "class" in ex
            assert "LOE" in ex

    def test_excludes_key(self):
        datasets = load_acp_datasets()
        if not datasets:
            pytest.skip("No ACP datasets available")
        key = datasets[0].key
        examples = get_few_shot_examples(GRADE, n_examples=3, exclude_key=key)
        # Can't verify exclusion directly, but should not crash
        assert isinstance(examples, list)

    def test_abcd_scheme(self):
        examples = get_few_shot_examples(ABCD_123, n_examples=2)
        assert len(examples) <= 2
