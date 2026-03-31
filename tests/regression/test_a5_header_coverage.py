from pathlib import Path

from ptodsl.lib import a5
from ptodsl.lib.a5.a5_header_coverage import (
    A5_HEADER_COVERAGE,
    A5_HEADER_INVENTORY,
    a5_header_coverage_markdown,
    a5_header_coverage_summary,
)


def test_a5_header_inventory_is_tracked_completely():
    assert len(A5_HEADER_INVENTORY) == 116
    assert set(A5_HEADER_COVERAGE) == set(A5_HEADER_INVENTORY)


def test_implemented_and_partial_a5_helpers_exist():
    for entry in A5_HEADER_COVERAGE.values():
        if entry["status"] not in {"implemented", "partial"}:
            continue
        helper = entry["helper"]
        if helper is None or "." in helper:
            continue
        assert getattr(a5, helper) is not None


def test_a5_header_coverage_markdown_mentions_all_headers():
    text = a5_header_coverage_markdown()
    for name in A5_HEADER_INVENTORY:
        assert f"`{name}`" in text


def test_a5_header_coverage_summary_matches_inventory():
    counts = a5_header_coverage_summary()
    assert sum(counts.values()) == len(A5_HEADER_INVENTORY)
    assert counts["implemented"] > 0
    assert counts["pending"] > 0


def test_checked_in_a5_header_checklist_is_in_sync():
    checklist = (
        Path(__file__).resolve().parents[2]
        / "ptodsl"
        / "lib"
        / "a5"
        / "A5_HEADER_COVERAGE.md"
    )
    assert checklist.read_text(encoding="utf-8") == a5_header_coverage_markdown()
