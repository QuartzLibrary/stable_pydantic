from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Self

import pydantic


class Imports(pydantic.BaseModel):
    decimal: bool = False
    datetime: bool = False
    date: bool = False
    time: bool = False
    timedelta: bool = False
    optional: bool = False
    union: bool = False
    uses_json: bool = False

    def touch(self, field_type: type[Any]):
        if field_type is type(None):
            pass
        elif field_type is bool:
            pass
        elif field_type is int:
            pass
        elif field_type is float:
            pass
        elif field_type is Decimal:
            self.decimal = True
        elif field_type is complex:
            pass
        elif field_type is str:
            pass
        elif field_type is bytes:
            pass
        elif field_type is bytearray:
            pass
        elif field_type is datetime:
            self.datetime = True
        elif field_type is date:
            self.date = True
        elif field_type is time:
            self.time = True
        elif field_type is timedelta:
            self.timedelta = True
        else:
            raise ValueError(f"Unsupported base type {field_type}")

    def imports(self) -> str:
        def filter(items: list[tuple[str, bool]]) -> list[str]:
            return [item[0] for item in items if item[1]]

        # import json
        # from datetime import date, datetime, time, timedelta
        # from decimal import Decimal
        # from typing import Optional, Union

        import_datetime = filter(
            [
                ("date", self.date),
                ("datetime", self.datetime),
                ("time", self.time),
                ("timedelta", self.timedelta),
            ]
        )
        import_decimal = filter(
            [
                ("Decimal", self.decimal),
            ]
        )
        import_typing = filter(
            [
                ("Optional", self.optional),
                ("Union", self.union),
            ]
        )

        imports = ""

        if self.uses_json:
            imports += "import json\n"
        if import_datetime:
            imports += f"from datetime import {', '.join(import_datetime)}\n"
        if import_decimal:
            imports += f"from decimal import {', '.join(import_decimal)}\n"
        if import_typing:
            imports += f"from typing import {', '.join(import_typing)}\n"

        if self.uses_json or import_datetime + import_decimal + import_typing:
            imports += "\n"

        return imports

    def combine(self, value: Self) -> "Imports":
        return Imports(
            decimal=self.decimal or value.decimal,
            datetime=self.datetime or value.datetime,
            date=self.date or value.date,
            time=self.time or value.time,
            timedelta=self.timedelta or value.timedelta,
            optional=self.optional or value.optional,
            union=self.union or value.union,
            uses_json=self.uses_json or value.uses_json,
        )
