from pathlib import Path

import pydantic

from stable_pydantic.model_graph import ModelNode


class CurrentEntry(pydantic.BaseModel):
    "The current version is only present for convenience, to get a nice diff."

    path: Path
    source: str

    @staticmethod
    def create(path: Path, model: ModelNode) -> "CurrentEntry":
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
    path: Path
    source: str

    @staticmethod
    def create(version: int, path: Path, source: str) -> "SchemaEntry":
        path.write_text(source)
        return SchemaEntry(
            version=version,
            path=path,
            source=source,
        )

    @staticmethod
    def open(version: int, path: Path) -> "SchemaEntry":
        return SchemaEntry(
            version=version,
            path=path,
            source=path.read_text(),
        )

    def assert_equal(self, model: ModelNode):
        assert self.source == model.clean_source_recursive(), (
            f"Schema at version {self.version} is not the same as live schema for model {model.__name__}."
        )


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

    def max_version(self) -> int | None:
        return max(self.versions.keys()) if self.versions else None

    def update_current(self):
        if self.current is None:
            self.current = CurrentEntry.create(self.path / "current.py", self.live)
        else:
            self.current.update(self.live)

    def assert_unchanged(self):
        assert self.current, (
            f"No current version present for model {self.live.node.__name__}."
        )
        self.current.assert_unchanged(self.live)
        max_version = self.max_version()
        if max_version is not None:
            self.versions[max_version].assert_equal(self.live)


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

    def assert_unchanged_schemas(self):
        for model in self.models.values():
            model.assert_unchanged()


def _read_schema_files(at: Path, model: type[pydantic.BaseModel]) -> ModelEntry:
    at = at / model.__name__
    at.mkdir(parents=True, exist_ok=True)

    current: CurrentEntry | None = None
    version: dict[int, SchemaEntry] = {}
    migrations: dict[tuple[int, int], MigrationEntry] = {}

    for file in at.glob("*.py"):
        number = _parse_schema_file_name(file.name)
        migration = _parse_schema_migration_file_name(file.name)
        if file.name == "__init__.py":
            pass
        elif file.name == "current.py":
            current = CurrentEntry.open(file)
        elif number is not None:
            version[number] = SchemaEntry.open(number, file)
        elif migration is not None:
            migrations[migration] = MigrationEntry.open(
                migration[0], migration[1], file
            )
        else:
            raise ValueError(f"Invalid schema file name: {file.name}")

    return ModelEntry(
        live=ModelNode.new(model),
        path=at,
        current=current,
        versions=version,
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
    if not name.startswith("schema_") or not name.endswith(".py"):
        return None

    from_part, to_part = name[len("schema_") : -len(".py")].split("_to_")
    try:
        return int(from_part), int(to_part)
    except ValueError:
        return None
