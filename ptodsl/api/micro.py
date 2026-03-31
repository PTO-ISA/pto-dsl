from mlir.dialects import pto as _pto


def _is_public_micro_symbol(name):
    if name.startswith("_"):
        return False
    if name in {
        "AddressSpace",
        "AddressSpaceAttr",
        "AlignType",
        "MaskType",
        "VRegType",
    }:
        return True
    if name.startswith("v") or name.startswith("p"):
        return True
    return False


def __getattr__(name):
    try:
        value = getattr(_pto, name)
    except AttributeError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    if not _is_public_micro_symbol(name):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return value


def __dir__():
    return sorted(__all__)


__all__ = sorted(name for name in dir(_pto) if _is_public_micro_symbol(name))
