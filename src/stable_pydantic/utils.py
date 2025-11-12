import typing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from types import UnionType

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

COMPOSITE_TYPES_EXT = {
    *COMPOSITE_TYPES,
    typing.Union,
    UnionType,
}
