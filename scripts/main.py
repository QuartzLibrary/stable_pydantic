import ast
import enum
import inspect
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import OrderedDict, Self

import mod
import pydantic


class Nope:
    pass


class A(pydantic.BaseModel):
    local: str
    local2: int


class Model(pydantic.BaseModel):
    local: A = pydantic.Field(
        default_factory=lambda: A(
            local="test",
            # aaa
            local2=10,
        )
    )

    remote: mod.A
    age: int

    ppp: int

    # nope: Nope

    def a(self) -> str:
        return "test"


_BASE_TYPES = {
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


class ModelNode(pydantic.BaseModel):
    node: type[pydantic.BaseModel]
    children: OrderedDict[str, Self]
    all: dict[type[pydantic.BaseModel], Self]

    alias: int | None = None
    """To deduplicate models that have the same name."""

    @staticmethod
    def new(model: type[pydantic.BaseModel]) -> "ModelNode":
        return _new(model)

    def ordered_nodes(self) -> list[Self]:
        nodes = []

        def _ordered_nodes(node: Self):
            if node in nodes:
                return
            nodes.append(node)
            for child in node.children.values():
                _ordered_nodes(child)

        _ordered_nodes(self)

        return nodes

    def recursive_source(self) -> str:
        """Generate source code for a Pydantic model with only its fields (no methods/properties)."""

        nodes = self.ordered_nodes()
        nodes.reverse()
        last = nodes[-1] if nodes else None

        source = ""
        for node in nodes:
            source += node.source()
            source += "\n"
            if node != last:
                source += "\n"

        return "import pydantic" + "\n\n" + source

    def source(self) -> str:
        name_map: dict[str, tuple[str, int]] = {
            field: (c.node.__name__, c.alias)
            for field, c in self.children.items()
            if c.alias is not None
        }

        name = self.node.__name__
        source = clean_model_source(self.node, name_map)

        return source.replace(
            f"class {name}(",
            f"class {name}_{self.alias}(",
        )


def _new(model: type[pydantic.BaseModel]) -> ModelNode:
    def _populate_aliases(self: ModelNode):
        all = {}

        for model, node in self.all.items():
            if model.__name__ not in all:
                all[model.__name__] = []

            all[model.__name__].append(node)

        duplicates = ((name, nodes) for name, nodes in all.items() if len(nodes) > 1)

        for _name, nodes in duplicates:
            for i, node in enumerate(nodes):
                node.alias = i

    all = {}

    def _new(model: type[pydantic.BaseModel]) -> ModelNode:
        if model in all:
            return all[model]

        children = OrderedDict()

        for _field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation

            assert isinstance(field_type, type)

            if field_type in _BASE_TYPES:
                continue
            elif field_type is enum.Enum:
                raise ValueError(f"Enum type {field_type} is todo")
            elif field_type is list:
                raise ValueError(f"List type {field_type} is todo")
            elif field_type is set:
                raise ValueError(f"Set type {field_type} is todo")
            elif field_type is tuple:
                raise ValueError(f"Tuple type {field_type} is todo")
            elif field_type is dict:
                raise ValueError(f"Dict type {field_type} is todo")
            elif issubclass(field_type, pydantic.BaseModel):
                children[_field_name] = _new(field_type)
            else:
                raise ValueError(f"Unsupported type {field_type}")

        node = ModelNode(
            node=model,
            children=children,
            all=all,
        )

        all[model] = node

        return node

    self = _new(model)
    _populate_aliases(self)
    return self


def clean_model_source(
    model: type[pydantic.BaseModel],
    name_map: dict[str, tuple[str, int]],
) -> str:
    source = inspect.getsource(model)
    tree = ast.parse(source)

    # Get the class definition (should be the first node)
    class_def = tree.body[0]
    if not isinstance(class_def, ast.ClassDef):
        raise ValueError("Expected a class definition")

    # Filter body to keep only annotated field definitions (AnnAssign nodes)
    new_body = []
    for node in class_def.body:
        # Field with type annotation like `name: str` or `name: str = value`
        if isinstance(node, ast.AnnAssign):
            # Get the field name (only handle simple names for now)
            if isinstance(node.target, ast.Name):
                field_name: str = node.target.id
                if field_name in name_map:
                    # Rename the type name in the annotation
                    class_name, alias = name_map[field_name]
                    node.annotation = _rename_annotation(
                        node.annotation, class_name, alias
                    )
            new_body.append(node)

    # If no fields, add pass statement
    if not new_body:
        new_body: list[ast.stmt] = [ast.Pass()]

    class_def.body = new_body

    # Convert back to source code
    return ast.unparse(tree)


def _rename_annotation(annotation: ast.expr, name: str, alias: int) -> ast.expr:
    """Recursively rename type names in an annotation according to name and alias."""
    aliased_name = _aliased_name(name, alias)

    if isinstance(annotation, ast.Name):
        # Simple name like `A` or `str`
        if annotation.id == name:
            annotation.id = aliased_name
    elif isinstance(annotation, ast.Attribute):
        # Attribute like `mod.A` - replace with just the aliased name (remove module prefix)
        if annotation.attr == name:
            return ast.Name(id=aliased_name, ctx=annotation.ctx)
    elif isinstance(annotation, ast.Subscript):
        # Generic types like `list[A]` or `dict[str, A]`
        annotation.slice = _rename_annotation(annotation.slice, name, alias)
    elif isinstance(annotation, ast.Tuple):
        # Tuple of types (e.g., in dict[str, A])
        annotation.elts = [
            _rename_annotation(elt, name, alias) for elt in annotation.elts
        ]
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # Union types like `A | B`
        annotation.left = _rename_annotation(annotation.left, name, alias)
        annotation.right = _rename_annotation(annotation.right, name, alias)
    else:
        raise ValueError(f"Unsupported annotation: {annotation}")

    return annotation


def _aliased_name(name: str, alias: int) -> str:
    """The aliased name for when two model names collide."""
    return f"{name}_{alias}"


def upgrade_current_schema(model: type[pydantic.BaseModel], at: Path | str):
    at = Path(at)
    at.mkdir(parents=True, exist_ok=True)

    node = ModelNode.new(model)

    live = node.recursive_source()

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

    live = node.recursive_source()

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
    print(name)
    print(number_part)
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
