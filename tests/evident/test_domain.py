"""Tests for evident.domain: enums serialize as plain strings, records are frozen."""

import dataclasses
from enum import Enum

import pytest

from evident import domain

_ENUMS = [v for v in vars(domain).values() if isinstance(v, type) and issubclass(v, Enum) and v.__module__ == domain.__name__]


@pytest.mark.parametrize("enum_cls", _ENUMS, ids=lambda e: e.__name__)
def test_enum_round_trips_through_its_string_value(enum_cls):
    for member in enum_cls:
        assert isinstance(member, str)
        assert enum_cls(member.value) is member


def test_records_are_frozen():
    rec = domain.RawRecommendation(ordinal=0, text="t", raw_strength="s", raw_certainty="c",
                                   raw_category=domain.Category.GRADED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.text = "x"
