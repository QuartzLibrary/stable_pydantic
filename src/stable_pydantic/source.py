import types
from pathlib import Path
from typing import Any

import pydantic

from stable_pydantic.model_graph import ModelNode


class CurrentEntry(pydantic.BaseModel):
    "The current version is only present for convenience, to get a nice diff."

    path: Path
    source: str

    @staticmethod
    def create(path: Path, model: ModelNode) -> "CurrentEntry":
        print(f"Creating current.py for {model.node.__name__}.")

        source = model.clean_source_recursive()
        path.write_text(source)
        return CurrentEntry(
            path=path,
            source=source,
        )

    @staticmethod
    def open(path: Path) -> "CurrentEntry":
        return CurrentEntry(
            path=path,
            source=path.read_text(),
        )

    def assert_unchanged(self, model: ModelNode):
        assert self.source == model.clean_source_recursive(), (
            f"Current schema is not the same as live schema for model {model.__name__}."
        )

    def update(self, model: ModelNode):
        new_source = model.clean_source_recursive()
        if self.source == new_source:
            print(f"current.py for model {model.node.__name__} is up to date.")
            return
        else:
            print(f"Updating current.py for model {model.node.__name__}.")
        self.source = new_source
        self.path.write_text(self.source)


class SchemaEntry(pydantic.BaseModel):
    version: int
    dir: Path
    source: str

    @staticmethod
    def create(dir: Path, version: int, model: ModelNode) -> "SchemaEntry":
        print(f"Creating schema at version {version} for {model.node.__name__}.")
        source = model.clean_source_recursive()
        (dir / f"schema_{version}.py").write_text(source)
        return SchemaEntry(
            version=version,
            dir=dir,
            source=source,
        )

    @staticmethod
    def open(dir: Path, version: int) -> "SchemaEntry":
        return SchemaEntry(
            version=version,
            dir=dir,
            source=(dir / f"schema_{version}.py").read_text(),
        )

    def assert_equal(self, model: ModelNode):
        assert self.source == model.clean_source_recursive(), (
            f"Schema at version {self.version} is not the same as live schema for model {model.__name__}."
        )

    def isolated_model(self) -> ModelNode:
        # Create a new module with a unique name
        module_name = f"schema_{self.version}"
        module = types.ModuleType(module_name)
        module.__file__ = str(self.dir / f"schema_{self.version}.py")

        # Execute the source code in the module's namespace
        exec(self.source, module.__dict__)

        # TODO: avoid this hack, but for now the largest must be the root.

        models = [
            obj
            for obj in module.__dict__.values()
            if isinstance(obj, type)
            and issubclass(obj, pydantic.BaseModel)
            and obj is not pydantic.BaseModel
        ]
        models.sort(key=lambda x: len(x.model_json_schema()), reverse=True)

        return ModelNode.new(models[0])

    def json_schema(self) -> dict[str, Any]:
        return self.isolated_model().model_json_schema()


class MigrationEntry(pydantic.BaseModel):
    from_version: int
    to_version: int
    path: Path
    source: str

    @staticmethod
    def create(
        from_version: int, to_version: int, path: Path, source: str
    ) -> "MigrationEntry":
        path.write_text(source)
        return MigrationEntry(
            from_version=from_version,
            to_version=to_version,
            path=path,
            source=source,
        )

    @staticmethod
    def open(from_version: int, to_version: int, path: Path) -> "MigrationEntry":
        return MigrationEntry(
            from_version=from_version,
            to_version=to_version,
            path=path,
            source=path.read_text(),
        )


class ModelEntry(pydantic.BaseModel):
    live: ModelNode

    path: Path
    current: CurrentEntry | None
    versions: dict[int, SchemaEntry]
    migrations: dict[tuple[int, int], MigrationEntry]

    @staticmethod
    def open(path: Path, model: type[pydantic.BaseModel]) -> "ModelEntry":
        return _read_schema_files(path, model)

    def latest_version(self) -> int | None:
        return max(self.versions.keys()) if self.versions else None

    def next_version(self) -> int:
        latest_version = self.latest_version()
        return latest_version + 1 if latest_version is not None else 0

    def update_current(self):
        if self.current is None:
            self.current = CurrentEntry.create(self.path / "current.py", self.live)
        else:
            self.current.update(self.live)

    def update_schemas(self, next_version: int):
        self.update_current()

        latest_version = self.latest_version()
        if (
            latest_version is not None
            and self.versions[latest_version].source
            == self.live.clean_source_recursive()
        ):
            print(
                f"{self.live.node.__name__} at version {latest_version} is up to date."
            )
            return

        self.versions[next_version] = SchemaEntry.create(
            self.path, next_version, self.live
        )

    def assert_unchanged(self):
        assert self.current, (
            f"No current version present for model {self.live.node.__name__}."
        )
        self.current.assert_unchanged(self.live)
        latest_version = self.latest_version()
        if latest_version is not None:
            self.versions[latest_version].assert_equal(self.live)

    def assert_compatible(self, forward: bool, backward: bool):
        assert forward or backward, "Assert compatibility in at least one direction."

        if not self.versions:
            return

        from stable_pydantic import compatibility

        versions = sorted(self.versions.values(), key=lambda x: x.version)

        print([v.version for v in versions])

        for i in range(len(versions) - 1):
            from_version = versions[i]
            to_version = versions[i + 1]
            compat = compatibility.check(from_version, to_version)
            if forward:
                assert compat.forward_compatible, (
                    f"For model {self.live.node.__name__}, version {from_version.version}"
                    + f" is not stricter than version {to_version.version}. Forward compatibility is not maintained."
                    + f"\nfrom_schema: {from_version.json_schema()}"
                    + f"\nto_schema: {to_version.json_schema()}"
                )
            if backward:
                assert compat.backward_compatible, (
                    f"For model {self.live.node.__name__}, version {to_version.version}"
                    + f" is not stricter than version {from_version.version}. Backward compatibility is not maintained."
                    + f"\nfrom_schema: {from_version.json_schema()}"
                    + f"\nto_schema: {to_version.json_schema()}"
                )


class SchemaFilesystem(pydantic.BaseModel):
    models: dict[type[pydantic.BaseModel], ModelEntry]

    @staticmethod
    def open(
        path: Path | str, models: list[type[pydantic.BaseModel]]
    ) -> "SchemaFilesystem":
        path = Path(path)
        return SchemaFilesystem(
            models={model: ModelEntry.open(path, model) for model in models}
        )

    def update_current(self):
        for model in self.models.values():
            model.update_current()
        self.assert_unchanged_schemas()

    def update_schemas(self):
        next_version = (
            max(model.next_version() for model in self.models.values())
            if self.models
            else 0
        )
        for model in self.models.values():
            print(f"Updating schemas for {model.live.node.__name__}.")
            model.update_schemas(next_version)
            next_version = max(next_version, model.next_version())

        self.assert_unchanged_schemas()

    def assert_unchanged_schemas(self):
        for model in self.models.values():
            model.assert_unchanged()

    def assert_compatible_schemas(self, forward: bool = False, backward: bool = True):
        """
        Assert that the schemas are compatible in the given direction.

        Forward compatible: old clients keep working with new data.
        Backward compatible: new clients keep working with old data.
        """
        for model in self.models.values():
            model.assert_compatible(forward, backward)


def _read_schema_files(at: Path, model: type[pydantic.BaseModel]) -> ModelEntry:
    at = at / model.__name__
    at.mkdir(parents=True, exist_ok=True)

    current: CurrentEntry | None = None
    versions: dict[int, SchemaEntry] = {}
    migrations: dict[tuple[int, int], MigrationEntry] = {}

    for file in at.glob("*.py"):
        number = _parse_schema_file_name(file.name)
        migration = _parse_schema_migration_file_name(file.name)
        if file.name == "__init__.py":
            pass
        elif file.name == "current.py":
            current = CurrentEntry.open(file)
        elif number is not None:
            versions[number] = SchemaEntry.open(at, number)
        elif migration is not None:
            migrations[migration] = MigrationEntry.open(
                migration[0], migration[1], file
            )
        else:
            raise ValueError(f"Invalid schema file name: {file.name}")

    for from_version, to_version in migrations.keys():
        assert from_version in versions
        assert to_version in versions

    return ModelEntry(
        live=ModelNode.new(model),
        path=at,
        current=current,
        versions=versions,
        migrations=migrations,
    )


def _parse_schema_file_name(name: str) -> int | None:
    if not name.startswith("schema_") or not name.endswith(".py"):
        return None
    number_part = name[len("schema_") : -len(".py")]
    try:
        return int(number_part)
    except ValueError:
        return None


def _parse_schema_migration_file_name(name: str) -> tuple[int, int] | None:
    if not name.startswith("schema_") or not name.endswith(".py") or "_to_" not in name:
        return None

    from_part, to_part = name[len("schema_") : -len(".py")].split("_to_")
    try:
        return int(from_part), int(to_part)
    except ValueError:
        return None
