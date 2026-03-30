from pathlib import Path

from ptodsl import tile
from ptodsl.lib import a5
from ptodsl.lib.a5.tile_micro_coverage import (
    TILE_MICRO_COVERAGE,
    coverage_markdown,
    coverage_summary,
)


def test_tile_micro_coverage_checklist_covers_every_tile_api_symbol():
    assert set(TILE_MICRO_COVERAGE) == set(tile.__all__)


def test_implemented_tile_micro_helpers_exist():
    for name, entry in TILE_MICRO_COVERAGE.items():
        helper = entry["helper"]
        if entry["status"] == "implemented":
            assert helper is not None
            assert getattr(a5, helper) is not None


def test_tile_micro_coverage_markdown_mentions_all_tile_ops():
    text = coverage_markdown()
    for name in tile.__all__:
        assert f"`{name}`" in text


def test_tile_micro_coverage_summary_matches_public_surface():
    counts = coverage_summary()
    assert sum(counts.values()) == len(tile.__all__)
    assert counts["implemented"] > 0
    assert counts["blocked"] > 0


def test_checked_in_tile_micro_checklist_is_in_sync():
    checklist = Path(__file__).resolve().parents[2] / "ptodsl" / "lib" / "a5" / "TILE_MICRO_CHECKLIST.md"
    assert checklist.read_text(encoding="utf-8") == coverage_markdown()
