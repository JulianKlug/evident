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
    GtMode,
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


def _write_icu_xlsx(path, doi, recs):
    pd.DataFrame({
        "Key": ["K"] + [None] * (len(recs) - 1),
        "Title": [f"Title {doi}"] + [None] * (len(recs) - 1),
        "DOI": [doi] + [None] * (len(recs) - 1),
        "recommendation": recs,
        "class": ["Conditional recommendation"] * len(recs),
        "LOE": ["low certainty of evidence"] * len(recs),
    }).to_excel(path, index=False)


def test_icu_loader_reads_all_xlsx(tmp_path, monkeypatch):
    import extraction.datasets as datasets_mod
    _write_icu_xlsx(tmp_path / "a.xlsx", "10.1/a", ["We suggest A", "We suggest A2"])
    _write_icu_xlsx(tmp_path / "b.xlsx", "10.1/b", ["We suggest B"])
    monkeypatch.setattr(datasets_mod, "_ICU_DIR", str(tmp_path))

    datasets = load_icu_datasets()

    assert sorted(ds.doi for ds in datasets) == ["10.1/a", "10.1/b"]
    assert sum(len(ds.ground_truth_df) for ds in datasets) == 3


def test_icu_loader_real_data_counts():
    datasets = load_icu_datasets()
    assert len(datasets) == 3
    assert sum(len(ds.ground_truth_df) for ds in datasets) == 17


def _labelled_grade_datasets():
    return load_acp_datasets() + load_icu_datasets()


class TestFewShotLeakageAndSeed:
    def test_no_own_gt_rows_when_excluded_by_doi(self):
        datasets = _labelled_grade_datasets()
        for ds in datasets:
            # Texts shared with another guideline are legitimately in the pool
            others = {t for o in datasets if o.doi != ds.doi for t in o.ground_truth_df["recommendation"]}
            own = set(ds.ground_truth_df["recommendation"]) - others

            examples = get_few_shot_examples(GRADE, n_examples=10_000, exclude_doi=f" https://doi.org/{ds.doi.upper()} ")

            assert own and not own & {ex["recommendation"] for ex in examples}, ds.doi

    def test_same_seed_same_examples(self):
        a = get_few_shot_examples(GRADE, n_examples=3, seed=42)
        b = get_few_shot_examples(GRADE, n_examples=3, seed=42)
        assert a == b

    def test_different_seed_different_examples(self):
        assert len(get_few_shot_examples(GRADE, n_examples=10_000)) >= 10
        a = get_few_shot_examples(GRADE, n_examples=3, seed=1)
        b = get_few_shot_examples(GRADE, n_examples=3, seed=2)
        assert a != b


class TestGtMode:
    @pytest.mark.parametrize("loader, n_rows", [(load_acp_datasets, 29), (load_icu_datasets, 17)])
    @pytest.mark.parametrize("mode", list(GtMode))
    def test_row_counts(self, loader, n_rows, mode):
        # ACP's single ungraded row (class "0") is already kept today, so both modes match
        assert sum(len(ds.ground_truth_df) for ds in loader(gt_mode=mode)) == n_rows

    def test_categories(self):
        datasets = load_acp_datasets(GtMode.WITH_UNGRADED) + load_icu_datasets(GtMode.WITH_UNGRADED)
        rows = pd.concat([ds.ground_truth_df for ds in datasets])
        no_rec = rows[rows["category"] == "no_recommendation"]["recommendation"].tolist()

        assert len(no_rec) == 2
        assert any(t.startswith("Evidence was inconclusive") for t in no_rec)
        assert set(rows["category"]) == {"graded", "no_recommendation"}

    def test_graded_only_has_no_category(self):
        assert "category" not in load_acp_datasets()[0].ground_truth_df.columns

    def test_raw_columns_equal_xlsx_cells(self):
        import os
        import extraction.datasets as datasets_mod
        ds = load_acp_datasets()[0]
        xlsx = pd.read_excel(os.path.join(datasets_mod._ACP_DIR, f"{ds.key}_extraction.xlsx")).dropna(subset=["recommendation"])

        assert ds.ground_truth_df["raw_class"].tolist() == xlsx["class"].astype(str).tolist()
        assert ds.ground_truth_df["raw_LOE"].tolist() == xlsx["LOE"].astype(str).tolist()
