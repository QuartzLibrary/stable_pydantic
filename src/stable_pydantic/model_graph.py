import ast
import enum
import inspect
import typing
from typing import Any, OrderedDict, Self

import pydantic

from stable_pydantic.imports import Imports
from stable_pydantic.utils import BASE_TYPES, COMPOSITE_TYPES_EXT


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
        imports = Imports()
        for node in nodes:
            new, new_touched_base_types = node.clean_source()
            source += new
            imports = imports.combine(new_touched_base_types)
            if node != last:
                source += "\n\n\n"

        return imports.imports() + "import pydantic" + "\n\n\n" + source + "\n"

    def clean_source(self) -> tuple[str, Imports]:
        from stable_pydantic import clean

        return clean.clean(self)

    def get_ast(self) -> ast.Module:
        return ast.parse(inspect.getsource(self.node))

    def get_class_ast(self) -> ast.ClassDef:
        tree = self.get_ast()
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.ClassDef)
        return tree.body[0]

    def aliased_name(self) -> str:
        # TODO: rename anything that might collide with primitive types.
        if self.alias is None:
            return self.node.__name__
        else:
            return f"{self.node.__name__}_{self.alias}"


def _new(model: type[pydantic.BaseModel]) -> ModelNode:
    def from_field_type(field_type: type[Any] | None) -> list[ModelNode] | None:
        if field_type is None or field_type in BASE_TYPES:
            return None

        elif field_type is enum.Enum:
            raise ValueError(f"Enum type {field_type} is todo")

        elif typing.get_origin(field_type) in COMPOSITE_TYPES_EXT:
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
