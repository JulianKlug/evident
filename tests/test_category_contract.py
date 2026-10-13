"""Parser category strings must equal evident's Category values; the store persists them as-is."""

from evident.domain import Category
from extraction.response_parser import CATEGORY_BEST_PRACTICE, CATEGORY_GRADED, CATEGORY_NO_RECOMMENDATION


def test_parser_categories_match_domain_enum():
    assert {CATEGORY_GRADED, CATEGORY_BEST_PRACTICE, CATEGORY_NO_RECOMMENDATION} == {c.value for c in Category}
