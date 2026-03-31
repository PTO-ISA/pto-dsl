import builtins


class ConstexprAnnotation:
    __ptodsl_constexpr__ = True

    def __init__(self, inner_type):
        self.inner_type = inner_type

    def __repr__(self):
        return f"Constexpr[{self.inner_type!r}]"


class Constexpr:
    def __class_getitem__(cls, inner_type):
        return ConstexprAnnotation(inner_type)


def is_constexpr_annotation(annotation):
    return getattr(annotation, "__ptodsl_constexpr__", False)


def const_expr(value):
    return value


def range_constexpr(*args):
    return builtins.range(*args)


__all__ = [
    "Constexpr",
    "ConstexprAnnotation",
    "const_expr",
    "is_constexpr_annotation",
    "range_constexpr",
]
