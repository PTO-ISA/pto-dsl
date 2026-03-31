from contextlib import contextmanager

from mlir.dialects import arith
from mlir.dialects import pto as _pto
from mlir.ir import IndexType, InsertionPoint

from .scalar import Value, _unwrap
from .type_def import SubTensorType, TensorType, TileBufType


def get_block_idx():
    return Value(_pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(_pto.GetSubBlockNumOp().result)


def get_block_num():
    return Value(_pto.GetBlockNumOp().result)


def _resolve_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        return _pto.LayoutAttr.get(getattr(_pto.Layout, layout))
    return layout


def _index_value(value):
    if isinstance(value, int):
        return arith.ConstantOp(IndexType.get(), value).result
    return _unwrap(value)


def _mul_index(lhs, rhs):
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs * rhs
    return Value(arith.MulIOp(_index_value(lhs), _index_value(rhs)).result)


def _resolve_tensor_dtype(type_or_value):
    candidate = type_or_value.type if hasattr(type_or_value, "type") else type_or_value
    return getattr(candidate, "element_type", None)


def _row_major_strides(shape):
    strides = [None] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        dim = shape[index]
        stride = _mul_index(dim, stride)
    return [_index_value(stride) for stride in strides]


class TensorView:
    def __init__(self, raw, *, dtype=None):
        self.raw = raw
        self.dtype = dtype if dtype is not None else _resolve_tensor_dtype(raw)

    def slice(self, offsets, sizes, *, static_shape=None):
        if static_shape is None:
            if not all(isinstance(size, int) for size in sizes):
                raise ValueError(
                    "TensorView.slice(...) requires static_shape when any size is dynamic."
                )
            static_shape = list(sizes)
        return TensorView(
            slice_view(
                SubTensorType(shape=static_shape, dtype=self.dtype),
                source=self.raw,
                offsets=offsets,
                sizes=sizes,
            ),
            dtype=self.dtype,
        )


class TileBufferSpec:
    def __init__(self, *, dtype, shape, space, valid_shape=None, config=None):
        self.dtype = dtype
        self.shape = list(shape)
        self.space = space
        self.valid_shape = list(shape) if valid_shape is None else list(valid_shape)
        self.config = config
        self._raw_type = TileBufType(
            shape=self.shape,
            dtype=dtype,
            memory_space=space,
            valid_shape=self.valid_shape,
            config=config,
        )

    @property
    def raw_type(self):
        return self._raw_type

    def alloc(self, *, addr=None, valid_row=None, valid_col=None):
        return TileBuffer(
            alloc_tile(
                self.raw_type,
                addr=addr,
                valid_row=valid_row,
                valid_col=valid_col,
            ),
            spec=self,
        )


class TileBuffer:
    def __init__(self, raw, *, spec=None):
        self.raw = raw
        self.spec = spec

    def load_from(self, view):
        load(view, self)
        return self

    def store_to(self, view):
        store(self, view)
        return view


def as_tensor(tensor_type, *, ptr, shape, strides, layout=None):
    shape_vals = [_unwrap(v) for v in shape]
    stride_vals = [_unwrap(v) for v in strides]
    kwargs = {}
    layout_attr = _resolve_layout_attr(layout)
    if layout_attr is not None:
        kwargs["layout"] = layout_attr
    return _pto.MakeTensorViewOp(
        tensor_type, _unwrap(ptr), shape_vals, stride_vals, **kwargs
    ).result


def slice_view(subtensor_type, *, source, offsets, sizes):
    offset_vals = [_index_value(v) for v in offsets]
    size_vals = [_index_value(v) for v in sizes]
    return _pto.PartitionViewOp(
        subtensor_type, _unwrap(source), offsets=offset_vals, sizes=size_vals
    ).result


@contextmanager
def vector_section():
    section = _pto.SectionVectorOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


@contextmanager
def cube_section():
    section = _pto.SectionCubeOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


def alloc_tile(tile_type, *, addr=None, valid_row=None, valid_col=None):
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _index_value(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _index_value(valid_col)
    return _pto.AllocTileOp(tile_type, **kwargs).result


def load(source, dest):
    _pto.TLoadOp(None, _unwrap(source), _unwrap(dest))


def store(source, dest):
    _pto.TStoreOp(None, _unwrap(source), _unwrap(dest))


def make_tensor(ptr, *, shape, strides=None, dtype=None, type=None, layout=None):
    if type is None:
        if dtype is None:
            raise ValueError("make_tensor(...) requires dtype when type is omitted.")
        type = TensorType(rank=len(shape), dtype=dtype)
    resolved_dtype = dtype if dtype is not None else _resolve_tensor_dtype(type)
    if strides is None:
        strides = _row_major_strides(shape)
    return TensorView(
        as_tensor(
            type,
            ptr=ptr,
            shape=[_index_value(v) for v in shape],
            strides=[_index_value(v) for v in strides],
            layout=layout,
        ),
        dtype=resolved_dtype,
    )


def make_tile_buffer(dtype, shape, *, space, valid_shape=None, config=None):
    return TileBufferSpec(
        dtype=dtype,
        shape=shape,
        space=space,
        valid_shape=valid_shape,
        config=config,
    )


def print(format, scalar):
    """
    Example:
    `print("hello %d\n", const(5))`
    is equivalent to
    `cce::printf("hello%d\n", 5);`

    NOTE: may not print if the print buffer is full from previous
    prints (typical when printing big tiles).
    """
    if isinstance(scalar, Value):
        scalar = _unwrap(scalar)

    _pto.print_(format, scalar)


__all__ = [
    "get_block_idx",
    "get_subblock_idx",
    "get_subblock_num",
    "get_block_num",
    "TensorView",
    "TileBuffer",
    "TileBufferSpec",
    "as_tensor",
    "make_tensor",
    "slice_view",
    "vector_section",
    "cube_section",
    "alloc_tile",
    "make_tile_buffer",
    "load",
    "store",
    "print",
]
