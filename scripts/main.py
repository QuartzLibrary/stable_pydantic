from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import mod
import pydantic

from stable_pydantic.model_graph import ModelNode


class Nope:
    pass


class Complete(pydantic.BaseModel):
    f_1: None
    f_2: bool
    f_3: int
    f_4: float
    f_5: complex = complex(1, 2)
    f_6: str
    f_7: bytes = b"test"
    f_8: datetime = datetime(2025, 12, 31, 12, 59, 59)
    f_9: date = date(2025, 12, 31)
    f_10: time = time(12, 59, 59)
    f_11: timedelta = timedelta(days=1, hours=2, minutes=3, seconds=4)
    f_12: Decimal = Decimal("1.23")
    f_13: list[int]
    f_14: set[int]
    f_15: tuple[int, int]
    f_16: dict[str, int]
    a: int | str
    f_17: Union[int, str]
    f_18: Optional[int]

    z: int | str | None | bool = 0


class A(pydantic.BaseModel):
    local: str = pydantic.Field()
    local2: int


class Model(pydantic.BaseModel):
    local: A = pydantic.Field(
        default_factory=lambda: A(
            local='te"st',
            local2=10,
        ),
        title="Local",
    )

    complete: Complete

    remote: mod.A
    age: int

    ppp: int

    # nope: Nope

    def a(self) -> str:
        return "test"


def upgrade_current_schema(model: type[pydantic.BaseModel], at: Path | str):
    at = Path(at)
    at.mkdir(parents=True, exist_ok=True)

    node = ModelNode.new(model)

    live = node.clean_source_recursive()

    filesystem = read_schema_files(at)
    if filesystem is None:
        write_new_version(model, 0, live, at)
        return

    (current_version, current_source, _version_files) = filesystem

    if current_source == live:
        print(
            f"Model {model.__name__} is up to date. "
            + f"Current version is {current_version}."
        )
        return

    write_new_version(model, current_version + 1, live, at)


def write_new_version(
    model: type[pydantic.BaseModel], version: int, live: str, at: Path
):
    (at / f"schema_{version}.py").write_text(live)
    (at / "current.py").write_text(live)

    print(f"Model {model.__name__} has been upgraded to version {version}.")


def test_current_schema(model: type[pydantic.BaseModel], at: Path | str):
    at = Path(at)
    at.mkdir(parents=True, exist_ok=True)

    node = ModelNode.new(model)

    live = node.clean_source_recursive()

    filesystem = read_schema_files(at)
    assert filesystem is not None, "No version present."
    (current_version, current_source, _version_files) = filesystem

    assert current_source == live, "Live schema is not the same as current schema"

    print(
        f"Model {model.__name__} is up to date. "
        + f"Current version is {current_version}."
    )


def read_schema_files(at: Path) -> tuple[int, str, dict[int, str]] | None:
    at.mkdir(parents=True, exist_ok=True)

    version_files: dict[int, str] = {}
    current: str | None = None

    for file in at.glob("*.py"):
        number = parse_schema_file_name(file.name)
        if file.name == "__init__.py":
            continue
        elif file.name == "current.py":
            current = file.read_text()
            continue
        elif number is not None:
            version_files[number] = file.read_text()
        else:
            raise ValueError(f"Invalid schema file name: {file.name}")

        with file.open("r") as f:
            version_files[number] = f.read()

    if current is None and not version_files:
        return None

    assert current is not None, "Expected a current schema file"

    max_version = max(version_files.keys())
    assert current == version_files[max_version], (
        "Current schema version is not the latest"
    )

    return max_version, current, version_files


def parse_schema_file_name(name: str) -> int | None:
    if not name.startswith("schema_") or not name.endswith(".py"):
        return None
    number_part = name[len("schema_") : -len(".py")]
    try:
        return int(number_part)
    except ValueError:
        return None


def main():
    upgrade_current_schema(Model, "./schemas")

    # node = ModelNode.new(Model)
    # print(node.recursive_source())


if __name__ == "__main__":
    main()
