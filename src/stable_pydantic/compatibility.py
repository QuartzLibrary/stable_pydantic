import enum
import typing
from types import UnionType
from typing import Any

import pydantic

from stable_pydantic.model_graph import ModelNode
from stable_pydantic.source import SchemaEntry
from stable_pydantic.utils import (
    BASE_TYPES,
    COMPOSITE_TYPES_EXT,
    UNION_TYPES,
    Unreachable,
    get_default,
    use,
)


class Compatibility(pydantic.BaseModel):
    forward_compatible: bool
    """Old clients keep working with new data."""
    backward_compatible: bool
    """New clients keep working with old data."""

    def combine(self, other: "Compatibility") -> "Compatibility":
        return Compatibility(
            forward_compatible=self.forward_compatible and other.forward_compatible,
            backward_compatible=self.backward_compatible and other.backward_compatible,
        )


COMPATIBLE = Compatibility(forward_compatible=True, backward_compatible=True)
INCOMPATIBLE = Compatibility(forward_compatible=False, backward_compatible=False)


def check(a: SchemaEntry, b: SchemaEntry) -> Compatibility:
    """Checks if a schema is more restrictive than another."""
    return _model_compatibility(a.isolated_model(), b.isolated_model())


def _model_compatibility(a: ModelNode, b: ModelNode) -> Compatibility:
    a_fields = {name: info for name, info in a.node.model_fields.items()}
    b_fields = {name: info for name, info in b.node.model_fields.items()}

    all_names = set(a_fields.keys()) | set(b_fields.keys())

    all_models = {model: node for model, node in a.all.items()} | {
        model: node for model, node in b.all.items()
    }

    compat = COMPATIBLE

    for name in all_names:
        a_field = a_fields.get(name)
        b_field = b_fields.get(name)

        inner_compat = None
        if a_field is None and b_field is None:
            raise ValueError(f"Field {name} is missing in both models")
        elif a_field is None:
            assert b_field is not None
            inner_compat = _new_field_compatibility(b_field, all_models)
        elif b_field is None:
            inner_compat = _dropped_field_compatibility(a_field, all_models)
        else:
            inner_compat = _field_compatibility(a_field, b_field, all_models)

        compat = compat.combine(inner_compat)

    return compat


def _new_field_compatibility(
    b: pydantic.fields.FieldInfo,
    all_models: dict[type[pydantic.BaseModel], ModelNode],
) -> Compatibility:
    return _annotation_compatibility(
        _or_none(b.annotation),
        b.annotation,
        all_models,
    )


def _dropped_field_compatibility(
    a: pydantic.fields.FieldInfo,
    all_models: dict[type[pydantic.BaseModel], ModelNode],
) -> Compatibility:
    return _annotation_compatibility(
        a.annotation,
        _or_none(a.annotation),
        all_models,
    )


def _field_compatibility(
    a: pydantic.fields.FieldInfo,
    b: pydantic.fields.FieldInfo,
    all_models: dict[type[pydantic.BaseModel], ModelNode],
) -> Compatibility:
    annotation_compatibility = _annotation_compatibility(
        a.annotation, b.annotation, all_models
    )

    default_compatibility = _default_compatibility(a, b)

    # TODO
    # alias: str | None
    # alias_priority: int | None
    # validation_alias: str | AliasPath | AliasChoices | None
    # serialization_alias: str | None
    # exclude: Whether to exclude the field from the model serialization.
    # exclude_if: A callable that determines whether to exclude a field during serialization based on its value.
    # json_schema_extra: A dict or callable to provide extra JSON schema properties.
    # 'strict': types.Strict,
    # 'gt': annotated_types.Gt,
    # 'ge': annotated_types.Ge,
    # 'lt': annotated_types.Lt,
    # 'le': annotated_types.Le,
    # 'multiple_of': annotated_types.MultipleOf,
    # 'min_length': annotated_types.MinLen,
    # 'max_length': annotated_types.MaxLen,
    # 'pattern': None,
    # 'allow_inf_nan': None,
    # 'max_digits': None,
    # 'decimal_places': None,
    # 'union_mode': None,
    # 'coerce_numbers_to_str': None,
    # 'fail_fast': types.FailFast,

    return annotation_compatibility.combine(default_compatibility)


def _default_compatibility(
    a: pydantic.fields.FieldInfo,
    b: pydantic.fields.FieldInfo,
) -> Compatibility:
    a_default = None
    a_has_default = True
    try:
        a_default = get_default(a)
    except ValueError:
        a_has_default = False

    b_default = None
    b_has_default = True
    try:
        b_default = get_default(b)
    except ValueError:
        b_has_default = False

    # TODO: which of these should be compatible?
    # Usually changing the default should not be a problem, but worth double-checking.
    match (a_has_default, b_has_default):
        case (True, True):
            pass
        case (True, False):
            pass
        case (False, True):
            pass
        case (False, False):
            return COMPATIBLE
        case _:
            raise Unreachable()

    use(a_default)
    use(b_default)

    return COMPATIBLE


def _annotation_compatibility(
    a: type[Any] | None,
    b: type[Any] | None,
    all_models: dict[type[pydantic.BaseModel], ModelNode],
) -> Compatibility:
    if a is None:
        raise ValueError(f"Field type is {a}.")

    elif b is None:
        raise ValueError(f"Field type is {b}.")

    elif typing.get_origin(a) in UNION_TYPES and typing.get_origin(b) in UNION_TYPES:
        subtypes_a = list(typing.get_args(a))
        subtypes_b = list(typing.get_args(b))

        assert len(subtypes_a) > 0
        assert len(subtypes_b) > 0

        a_compats_first = [
            [
                _annotation_compatibility(subtype_a, subtype_b, all_models)
                for subtype_b in subtypes_b
            ]
            for subtype_a in subtypes_a
        ]

        b_compats_first = [
            [
                _annotation_compatibility(subtype_a, subtype_b, all_models)
                for subtype_a in subtypes_a
            ]
            for subtype_b in subtypes_b
        ]

        # All new subtypes must be compatible with at least one old subtype to maintain forward compatibility.
        forward_compatible = all(
            any(compat.forward_compatible for compat in compats)
            for compats in b_compats_first
        )

        # All old subtypes must be compatible with at least one new subtype to maintain backward compatibility.
        backward_compatible = all(
            any(compat.backward_compatible for compat in compats)
            for compats in a_compats_first
        )

        assert forward_compatible is not None
        assert backward_compatible is not None

        return Compatibility(
            forward_compatible=forward_compatible,
            backward_compatible=backward_compatible,
        )

    elif typing.get_origin(a) in UNION_TYPES:
        subtypes = list(typing.get_args(a))
        compats = [
            _annotation_compatibility(subtype, b, all_models) for subtype in subtypes
        ]
        return Compatibility(
            # At least one old subtype must be compatible with the new one to maintain forward compatibility.
            forward_compatible=any(compat.forward_compatible for compat in compats),
            # All old subtypes must be compatible with the new one to maintain backward compatibility.
            backward_compatible=all(compat.backward_compatible for compat in compats),
        )

    elif typing.get_origin(b) in UNION_TYPES:
        subtypes = list(typing.get_args(b))
        compats = [
            _annotation_compatibility(a, subtype, all_models) for subtype in subtypes
        ]
        return Compatibility(
            # All new subtypes must be compatible with the old one to maintain forward compatibility.
            forward_compatible=all(compat.forward_compatible for compat in compats),
            # At least one new subtype must be compatible with the old one to maintain backward compatibility.
            backward_compatible=any(compat.backward_compatible for compat in compats),
        )

    elif a in BASE_TYPES and b in BASE_TYPES:
        # TODO: upgrades like int -> float
        if a is b:
            return COMPATIBLE
        else:
            return INCOMPATIBLE

    elif a in BASE_TYPES or b in BASE_TYPES:
        return INCOMPATIBLE

    elif a is enum.Enum or b is enum.Enum:
        raise ValueError("Enum type is todo")

    elif (
        typing.get_origin(a) in COMPOSITE_TYPES_EXT
        and typing.get_origin(b) in COMPOSITE_TYPES_EXT
    ):
        if typing.get_origin(a) is not typing.get_origin(b):
            return INCOMPATIBLE

        compats = [
            _annotation_compatibility(a, b, all_models)
            for a, b in zip(typing.get_args(a), typing.get_args(b), strict=True)
        ]
        return Compatibility(
            forward_compatible=all(compat.forward_compatible for compat in compats),
            backward_compatible=all(compat.backward_compatible for compat in compats),
        )

    elif issubclass(a, pydantic.BaseModel) and issubclass(b, pydantic.BaseModel):
        return _model_compatibility(all_models[a], all_models[b])

    else:
        raise ValueError(f"Unsupported type {a} or {b}")


def _or_none(v: type[Any] | None) -> type[Any] | None:
    if v is None:
        return None

    union: UnionType = v | type(None)
    union_: type[Any] = union  # type: ignore
    return union_
