from pathlib import Path

from ptodsl.lib import a5


_GENERATED_DIR = (
    Path(__file__).resolve().parents[2]
    / "ptodsl"
    / "lib"
    / "a5"
    / "generated"
    / "tile_ops"
)


def test_generated_tile_op_pto_files_exist_and_show_micro_ops():
    for op_name, spec in a5.TILE_OP_KERNEL_SPECS.items():
        text = (_GENERATED_DIR / f"{op_name}.pto").read_text(encoding="utf-8")
        assert f"func.func @tile_op_{op_name}" in text
        assert "pto.section.vector" in text
        for token in spec["expected_tokens"]:
            assert token in text


def test_tile_op_generation_index_covers_public_tile_ops():
    text = (_GENERATED_DIR / "TILE_OP_GENERATION_INDEX.md").read_text(encoding="utf-8")
    for op_name in a5.TILE_MICRO_COVERAGE:
        assert f"`{op_name}`" in text
    assert "`matmul` | `blocked`" in text
    assert "`subset` | `not_applicable`" in text
