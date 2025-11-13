import inspect
import typing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from types import UnionType
from typing import Any, Self

import pydantic

BASE_TYPES = {
    type(None),
    bool,
    int,
    float,
    Decimal,
    complex,
    str,
    bytes,
    bytearray,
    datetime,
    date,
    time,
    timedelta,
}

COMPOSITE_TYPES = {
    list,
    set,
    tuple,
    dict,
}

UNION_TYPES = {
    typing.Union,
    UnionType,
}

COMPOSITE_TYPES_EXT = {
    *COMPOSITE_TYPES,
    *UNION_TYPES,
}


def get_default(
    v: pydantic.fields.FieldInfo,
) -> Any:
    # Does this hold?
    assert v.default_factory is not pydantic.fields.PydanticUndefined

    if v.default_factory is not None:
        assert v.default is pydantic.fields.PydanticUndefined

        # TODO: how to get the validated data?
        assert 0 == len(inspect.signature(v.default_factory).parameters), (
            "Default factory using the validated data is not currently supported."
        )

        return v.default_factory()  # type: ignore (can't tell we asserted no parameters)
    elif v.default is not pydantic.fields.PydanticUndefined:
        assert v.default_factory is None
        return v.default
    else:
        assert v.default is pydantic.fields.PydanticUndefined
        assert v.default_factory is None
        raise ValueError(f"Field {v} has no default")


class Unreachable(Exception):
    def __init__(self: Self) -> None:
        super().__init__(
            "Unreachable: a unreachable code branch has been reached. This is a bug."
        )


def use(_v: Any):
    pass
