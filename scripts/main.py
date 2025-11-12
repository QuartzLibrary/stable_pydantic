import ast
import enum
import inspect
import typing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Callable, Generator, Optional, OrderedDict, Self, Union

import mod
import pydantic


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

_COMPOSITE_TYPES = {
    list,
    set,
    tuple,
    dict,
}

_COMPOSITE_TYPES_EXT = {
    *_COMPOSITE_TYPES,
    typing.Union,
    UnionType,
}


class TouchedBaseTypes(pydantic.BaseModel):
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

    def combine(self, value: Self) -> "TouchedBaseTypes":
        return TouchedBaseTypes(
            decimal=self.decimal or value.decimal,
            datetime=self.datetime or value.datetime,
            date=self.date or value.date,
            time=self.time or value.time,
            timedelta=self.timedelta or value.timedelta,
            optional=self.optional or value.optional,
            union=self.union or value.union,
            uses_json=self.uses_json or value.uses_json,
        )


class ModelNode(pydantic.BaseModel):
    node: type[pydantic.BaseModel]
    fields: OrderedDict[str, list[Self]]
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
            for field in node.fields.values():
                for child in field:
                    _ordered_nodes(child)

        _ordered_nodes(self)

        return nodes

    def clean_source_recursive(self) -> str:
        """Generate source code for a Pydantic model with only its fields (no methods/properties)."""

        nodes = self.ordered_nodes()
        nodes.reverse()
        last = nodes[-1] if nodes else None

        source = ""
        touched_base_types = TouchedBaseTypes()
        for node in nodes:
            new, new_touched_base_types = node.clean_source()
            source += new
            touched_base_types = touched_base_types.combine(new_touched_base_types)
            if node != last:
                source += "\n\n\n"

        return (
            touched_base_types.imports() + "import pydantic" + "\n\n\n" + source + "\n"
        )

    def clean_source(self) -> tuple[str, TouchedBaseTypes]:
        tree = self.ast()
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.ClassDef)
        _keep_only_class_fields(tree)
        assert len(
            [v for v in tree.body[0].body if not isinstance(v, ast.Pass)]
        ) == len(self.node.model_fields)
        touched_base_types = clean(self, tree)
        source = ast.unparse(tree)
        name = self.node.__name__
        return source.replace(
            f"class {name}(",
            f"class {self.aliased_name()}(",
        ), touched_base_types

    def ast(self) -> ast.Module:
        return ast.parse(inspect.getsource(self.node))

    def aliased_name(self) -> str:
        # TODO: rename anything that might collide with primitive types.
        if self.alias is None:
            return self.node.__name__
        else:
            return f"{self.node.__name__}_{self.alias}"


def _new(model: type[pydantic.BaseModel]) -> ModelNode:
    def from_field_type(field_type: type[Any] | None) -> list[ModelNode] | None:
        if field_type is None or field_type in _BASE_TYPES:
            return None

        elif field_type is enum.Enum:
            raise ValueError(f"Enum type {field_type} is todo")

        elif typing.get_origin(field_type) in _COMPOSITE_TYPES_EXT:
            values = []
            for value in typing.get_args(field_type):
                new = from_field_type(value)
                if new:
                    values.append(new)
            return values if values else None

        elif not isinstance(field_type, type):
            raise ValueError(f"Unsupported value {field_type}")

        elif issubclass(field_type, pydantic.BaseModel):
            return [_new(field_type)]

        else:
            raise ValueError(f"Unsupported type {field_type}")

    def from_field_info(info: pydantic.fields.FieldInfo) -> list[ModelNode] | None:
        return from_field_type(info.annotation)

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
            field = from_field_info(field_info)
            if field is None:
                continue
            else:
                children[_field_name] = field

        node = ModelNode(
            node=model,
            fields=children,
            all=all,
        )

        all[model] = node

        return node

    self = _new(model)
    _populate_aliases(self)
    return self


def clean(model: ModelNode, tree: ast.Module) -> TouchedBaseTypes:
    touched_base_types = TouchedBaseTypes()

    def unpack_composite_annotation(
        annotation: ast.Subscript,
    ) -> list[tuple[ast.expr, Callable[[ast.expr], None]]]:
        """Extract inner type annotations from subscripted composite types like list[int] or dict[str, int]."""
        slice_node = annotation.slice
        if isinstance(slice_node, ast.Tuple):
            # Multiple type parameters like dict[str, int] or tuple[int, str]

            def replace_multi(a: ast.expr, i: int):
                slice_node.elts[i] = a

            return [
                (elt, lambda a: replace_multi(a, i))
                for i, elt in enumerate(slice_node.elts)
            ]
        else:
            # Single type parameter like list[int] or set[str]
            def replace(a: ast.expr):
                annotation.slice = a

            return [(slice_node, replace)]

    def clean_field_type(
        field_type: type[Any] | None,
        annotation: ast.expr,
        replace: Callable[[ast.expr], None],
    ):
        if field_type is None:
            raise ValueError(f"Field type is None: {annotation}")

        elif field_type in _BASE_TYPES:
            touched_base_types.touch(field_type)
            # TODO: assert the types match.
            pass

        elif field_type is enum.Enum:
            raise ValueError(f"Enum type {field_type} is todo")

        elif typing.get_origin(field_type) is UnionType:

            def iter_binop(
                node: ast.BinOp,
            ) -> Generator[tuple[ast.expr, Callable[[ast.expr], None]], None, None]:
                def replace_left(a: ast.expr):
                    node.left = a

                def replace_right(a: ast.expr):
                    node.right = a

                if isinstance(node.left, ast.BinOp):
                    assert isinstance(node.left.op, ast.BitOr)
                    yield from iter_binop(node.left)
                else:
                    yield node.left, replace_left

                yield node.right, replace_right

            field_types = list(typing.get_args(field_type))

            assert isinstance(annotation, ast.BinOp)
            assert isinstance(annotation.op, ast.BitOr)

            for inner_field_type, (inner_annotation, replace) in zip(
                field_types, iter_binop(annotation), strict=True
            ):
                clean_field_type(inner_field_type, inner_annotation, replace)

        elif typing.get_origin(field_type) is typing.Union:
            field_types = list(typing.get_args(field_type))

            assert isinstance(annotation, ast.Subscript)
            annotations = unpack_composite_annotation(annotation)
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Optional"
            ):
                # Optional[T] is lowered to Union[T, None], so the number of parameters differs.
                assert len(field_types) == 2
                assert len(annotations) == 1
                field_types.remove(NoneType)
                assert len(field_types) == 1
                touched_base_types.optional = True
            elif (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Union"
            ):
                touched_base_types.union = True
            else:
                raise ValueError(f"Unsupported union type {annotation}")

            for inner_field_type, (inner_annotation, replace) in zip(
                field_types, annotations, strict=True
            ):
                clean_field_type(inner_field_type, inner_annotation, replace)

        elif typing.get_origin(field_type) in _COMPOSITE_TYPES:
            field_types = list(typing.get_args(field_type))

            assert isinstance(annotation, ast.Subscript)
            annotations = unpack_composite_annotation(annotation)

            for inner_field_type, (inner_annotation, replace) in zip(
                field_types, annotations, strict=True
            ):
                clean_field_type(inner_field_type, inner_annotation, replace)

        elif not isinstance(field_type, type):
            raise ValueError(f"Unsupported value {field_type}")

        elif issubclass(field_type, pydantic.BaseModel):
            inline_model = model.all[field_type]
            if inline_model.alias is not None:
                aliased_name = inline_model.aliased_name()
                if isinstance(annotation, ast.Name):
                    # Modify the annotation to use the aliased name
                    annotation.id = aliased_name
                elif isinstance(annotation, ast.Attribute):
                    replace(ast.Name(id=aliased_name, ctx=annotation.ctx))
                else:
                    raise ValueError(
                        f"Expected name or attribute annotation, got: {ast.dump(annotation)}"
                    )

        else:
            raise ValueError(f"Unsupported type {field_type}")

        return

    def clean_field_default(
        default: Any,
        default_factory: Callable[[], Any] | Callable[[dict[str, Any]], Any] | None,
        value: ast.expr,
    ) -> ast.expr | None:
        """Clean field default values by serializing to JSON and replacing with json.loads()."""
        import json

        actual_default: Any
        if default_factory is not None:
            assert default is pydantic.fields.PydanticUndefined

            # TODO: how to get the validated data?
            assert 0 == len(inspect.signature(default_factory).parameters), (
                "Default factory using the validated data is not currently supported."
            )
            actual_default = default_factory()  # type: ignore (can't tell we asserted no parameters)
        elif default is not pydantic.fields.PydanticUndefined:
            assert default_factory is None
            actual_default = default
        else:
            return None

        if isinstance(
            actual_default, (type(None), bool, int, float, str, bytes, complex, Decimal)
        ):
            new_value = ast.Constant(value=actual_default)
        elif isinstance(actual_default, (datetime, date, time, timedelta)):
            stringified_constant = ast.unparse(ast.Constant(value=actual_default))
            assert stringified_constant.startswith("datetime.")
            reparsed_module = ast.parse(stringified_constant[len("datetime.") :])
            assert len(reparsed_module.body) == 1
            reparsed_constant = reparsed_module.body[0].value  # type: ignore
            assert isinstance(reparsed_constant, ast.Call)
            new_value = reparsed_constant
        elif isinstance(actual_default, pydantic.BaseModel):
            # For Pydantic models, serialize to JSON and load inline with model_validate_json
            model_type = type(actual_default)
            model_node = model.all.get(model_type)
            assert model_node is not None, (
                f"Model type {model_type} not found in the model tree."
            )

            # ModelName.model_validate_json("...")
            new_value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=model_node.aliased_name(), ctx=ast.Load()),
                    attr="model_validate_json",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=actual_default.model_dump_json())],
                keywords=[],
            )
        else:
            touched_base_types.uses_json = True

            # json.loads("...")
            new_value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="json", ctx=ast.Load()),
                    attr="loads",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=json.dumps(actual_default))],
                keywords=[],
            )

        # Check if value is a Field() call, and if so update its arguments
        if isinstance(value, ast.Call) and (
            (isinstance(value.func, ast.Attribute) and value.func.attr == "Field")
            or (isinstance(value.func, ast.Name) and value.func.id == "Field")
        ):
            # Replace 'default' or 'default_factory' keyword arguments
            modified = False

            for keyword in value.keywords:
                if keyword.arg == "default_factory":
                    assert not modified, "Multiple default or default_factory keywords"
                    modified = True

                    keyword.value = ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=new_value,
                    )
                elif keyword.arg == "default":
                    assert not modified, "Multiple default or default_factory keywords"
                    modified = True

                    keyword.value = new_value

            assert modified, "No default or default_factory keyword found"

            return value
        else:
            return new_value

    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.ClassDef)

    for raw in tree.body[0].body:
        if isinstance(raw, ast.Pass):
            continue

        assert isinstance(raw, ast.AnnAssign)
        ast_node: ast.AnnAssign = raw
        assert isinstance(ast_node.target, ast.Name)

        field_name: str = ast_node.target.id
        model_field = model.node.model_fields[field_name]

        def replace(a: ast.expr):
            ast_node.annotation = a

        clean_field_type(model_field.annotation, ast_node.annotation, replace)

        if ast_node.value is not None:
            cleaned_default = clean_field_default(
                model_field.default,
                model_field.default_factory,
                ast_node.value,
            )
            if cleaned_default is not None:
                ast_node.value = cleaned_default

    return touched_base_types


def _keep_only_class_fields(tree: ast.Module):
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.ClassDef)

    new_body = []
    for node in tree.body[0].body:
        if isinstance(node, ast.AnnAssign):
            # Field with type annotation like `name: str` or `name: str = value`
            new_body.append(node)
        elif isinstance(node, ast.Pass):
            continue
        elif isinstance(node, ast.FunctionDef):
            continue
        else:
            raise ValueError(f"Unsupported node: {node}\n{ast.dump(node)}")

    if not new_body:
        # If no fields, add pass statement
        new_body: list[ast.stmt] = [ast.Pass()]

    tree.body[0].body = new_body


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
