from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Optional, Union

import pydantic


class A_1(pydantic.BaseModel):
    remote: str
    remote_2: int


class Complete(pydantic.BaseModel):
    f_1: None
    f_2: bool
    f_3: int
    f_4: float
    f_5: complex
    f_6: str
    f_7: bytes
    f_8: datetime
    f_9: date
    f_10: time
    f_11: timedelta
    f_12: Decimal
    f_13: list[int]
    f_14: set[int]
    f_15: tuple[int, int]
    f_16: dict[str, int]
    a: int | str
    f_17: Union[int, str]
    f_18: Optional[int]
    z: int | str | None | bool = 0


class A_0(pydantic.BaseModel):
    local: str
    local2: int


class Model(pydantic.BaseModel):
    local: A_0 = pydantic.Field(default_factory=lambda: A(local='test', local2=10))
    complete: Complete
    remote: A_1
    age: int
    ppp: int
