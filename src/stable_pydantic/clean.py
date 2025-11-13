import ast
import enum
import inspect
import typing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from types import NoneType, UnionType
from typing import Any, Callable, Generator

import pydantic

from stable_pydantic.imports import Imports
from stable_pydantic.model_graph import ModelNode
from stable_pydantic.utils import BASE_TYPES, COMPOSITE_TYPES


def clean(model: ModelNode) -> tuple[str, Imports]:
    tree = model.get_ast()
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.ClassDef)
    keep_only_class_fields(tree.body[0])
    assert len([v for v in tree.body[0].body if not isinstance(v, ast.Pass)]) == len(
        model.node.model_fields
    )
    touched_base_types = _clean(model, tree.body[0])
    source = ast.unparse(tree)
    name = model.node.__name__
    return source.replace(
        f"class {name}(",
        f"class {model.aliased_name()}(",
    ), touched_base_types


def keep_only_class_fields(class_def: ast.ClassDef):
    new_body = []
    for node in class_def.body:
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

    class_def.body = new_body


def _clean(model: ModelNode, tree: ast.ClassDef) -> Imports:
    imports = Imports()

    for raw in tree.body:
        if isinstance(raw, ast.Pass):
            continue

        assert isinstance(raw, ast.AnnAssign)
        ast_node: ast.AnnAssign = raw
        assert isinstance(ast_node.target, ast.Name)

        field_name: str = ast_node.target.id
        model_field = model.node.model_fields[field_name]

        def replace(a: ast.expr):
            ast_node.annotation = a

        _clean_field_type(
            model_field.annotation,
            ast_node.annotation,
            replace,
            model.all,
            imports,
        )

        if ast_node.value is not None:
            cleaned_default = _clean_field_default(
                model_field.default,
                model_field.default_factory,
                ast_node.value,
                model.all,
                imports,
            )
            if cleaned_default is not None:
                ast_node.value = cleaned_default

    return imports


def _clean_field_type(
    field_type: type[Any] | None,
    annotation: ast.expr,
    replace: Callable[[ast.expr], None],
    all: dict[type[pydantic.BaseModel], ModelNode],
    imports: Imports,
):
    if field_type is None:
        raise ValueError(f"Field type is None: {annotation}")

    elif field_type in BASE_TYPES:
        imports.touch(field_type)
        # TODO: assert the types match.
        pass

    elif field_type is enum.Enum:
        raise ValueError(f"Enum type {field_type} is todo")

    elif typing.get_origin(field_type) is UnionType:
        field_types = list(typing.get_args(field_type))

        assert isinstance(annotation, ast.BinOp)
        assert isinstance(annotation.op, ast.BitOr)

        for inner_field_type, (inner_annotation, replace) in zip(
            field_types, _iter_binop(annotation), strict=True
        ):
            _clean_field_type(inner_field_type, inner_annotation, replace, all, imports)

    elif typing.get_origin(field_type) is typing.Union:
        field_types = list(typing.get_args(field_type))

        assert isinstance(annotation, ast.Subscript)
        annotations = _unpack_composite_annotation(annotation)
        if isinstance(annotation.value, ast.Name) and annotation.value.id == "Optional":
            # Optional[T] is lowered to Union[T, None], so the number of parameters differs.
            assert len(field_types) == 2
            assert len(annotations) == 1
            field_types.remove(NoneType)
            assert len(field_types) == 1
            imports.optional = True
        elif isinstance(annotation.value, ast.Name) and annotation.value.id == "Union":
            imports.union = True
        else:
            raise ValueError(f"Unsupported union type {annotation}")

        for inner_field_type, (inner_annotation, replace) in zip(
            field_types, annotations, strict=True
        ):
            _clean_field_type(inner_field_type, inner_annotation, replace, all, imports)

    elif typing.get_origin(field_type) in COMPOSITE_TYPES:
        field_types = list(typing.get_args(field_type))

        assert isinstance(annotation, ast.Subscript)
        annotations = _unpack_composite_annotation(annotation)

        for inner_field_type, (inner_annotation, replace) in zip(
            field_types, annotations, strict=True
        ):
            _clean_field_type(inner_field_type, inner_annotation, replace, all, imports)

    elif issubclass(field_type, pydantic.BaseModel):
        inline_model = all[field_type]
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


def _clean_field_default(
    default: Any,
    default_factory: Callable[[], Any] | Callable[[dict[str, Any]], Any] | None,
    value: ast.expr,
    all: dict[type[pydantic.BaseModel], ModelNode],
    imports: Imports,
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
        model_node = all.get(model_type)
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
        imports.uses_json = True

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


def _unpack_composite_annotation(
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


def _iter_binop(
    node: ast.BinOp,
) -> Generator[tuple[ast.expr, Callable[[ast.expr], None]], None, None]:
    def replace_left(a: ast.expr):
        node.left = a

    def replace_right(a: ast.expr):
        node.right = a

    if isinstance(node.left, ast.BinOp):
        assert isinstance(node.left.op, ast.BitOr)
        yield from _iter_binop(node.left)
    else:
        yield node.left, replace_left

    yield node.right, replace_right
