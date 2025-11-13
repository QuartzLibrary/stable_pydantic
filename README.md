# stable_pydantic

`stable_pydantic` helps you avoid breaking changes and navigate schema changes of your `pydantic` models.

> ⚠️ This README.md describes the intended final product of this repo, but we aren't there yet. It's also likely to change as work on it progresses.

TODO:
- [x] Reference files
- [ ] Compatibility check
- [ ] Manual migrations

Minor TODOs:
- [ ] Prefix all models to avoid polluting grep-space.
- [ ] Expand to more types
- [ ] Systematically go through all pydantic knobs to check for relevance.

Unsolved questions:
- Are we happy treating `None` and absent fields interchangeably? It's almost always right, but in rare cases they might mean different things.

## Levels of stability

### Level 1: reference files

At this level, you just dump your schema to a file and check it into git. This allows you to see all important schema changes in one place.

Folder structure:

```
schemas/
  Model1/current.py
  Model2/current.py
```

```python
# test.py

MODELS = [Model1, Model2]

def test_schemas():
    stable_pydantic.skip_if_migrating()
  
    # Assert that the current schema is unchanged
    stable_pydantic.assert_unchanged_schemas("./schemas", MODELS)

def test_regenerate_schemas(request):
    # To run:
    # pytest -m stable_pydantic
    stable_pydantic.skip_if_not_migrating()

    # Overwrite the schema with the new one.
    stable_pydantic.regenerate_schemas("./schemas", MODELS)
```

### Level 2: backward compatible changes only

Here multiple schema versions are checked into git, and compatibility between them is automatically checked.
You will always be able to deserialize old data with your latest model.

> I'll review these lists later, just jotting down what comes to mind as I go here.

Backward compatible changes for json are:
- Reorder fields
- Add optional fields (equivalent to `None` -> `None | int`)
- Switch to a more generic type, but not vice versa.
  - `int` -> `float`
  - `int` -> `int | str`
  - `int` -> `int | None`
- Drop fields (equivalent to making them optional and always setting them to `None`)

Forward compatible (revertable or old clients) changes for json are:
- Reorder fields
- Drop optional fields
- Add optional fields



```
schemas/
  Model1/
    current.py
    schema_0.py
  Model2/
    current.py
    schema_1.py # Version numbers are unique across models.
```

```python
# test.py
MODELS = [Model1, Model2]

def test_schemas():
    stable_pydantic.skip_if_migrating()
  
    # Assert that all the schemas are compatible
    stable_pydantic.assert_compatible_schemas("./schemas", MODELS)
    # Assert that the current schema is unchanged
    stable_pydantic.assert_unchanged_schemas("./schemas", MODELS)

def test_update_schemas(request):
    # To run:
    # pytest -m stable_pydantic
    stable_pydantic.skip_if_not_migrating()
  
    # Add the new schema
    stable_pydantic.update_schemas("./schemas", MODELS)
```

### Level 3: Turing's migration

Here arbitrary migration logic is allowed, including handwritten logic. The mock models that have been checked into git are used to deserialize and then stage-by-stage update old values. This will require including a single (!) version number in the serialized data.

> With greater power comes greater complexity. Do you need it?

```
schemas/
  Model1/
    current.py
    schema_0.py
    schema_0_to_3.py
    schema_3.py
  Model2/
    current.py
    schema_1.py # Version numbers are unique across models.
    schema_4.py # If no upgrade file is present, it is assumed to be backward compatible.
```

```python
# test.py
MODELS = [Model1, Model2]

def test_schemas():
    stable_pydantic.skip_if_migrating()
  
    # Assert that all the schemas without a manual migration are compatible
    stable_pydantic.assert_compatible_schemas("./schemas", MODELS, manual_migrations=True)
    # Assert that the current schema is unchanged
    stable_pydantic.assert_unchanged_schemas("./schemas", MODELS)

def test_update_schemas(request):
    # To run:
    # pytest -m stable_pydantic
    stable_pydantic.skip_if_not_migrating()
  
    # Add the new schema
    stable_pydantic.update_schemas("./schemas", MODELS)

# A migration file can be added if the latest changes are not trivially compatible.
```

```python
# app.py
MODEL1_VERSION_HANDLER = stable_pydantic.version_handler(Model1)

old_json = "<json>"
value = Model1(...)

value_json: str = MODEL1_VERSION_HANDLER.to_json(value) # Handles injecting a version number.
old: Model1 = MODEL1_VERSION_HANDLER.from_json(old_data) # Handles unpacking and upgrading old versions.
```
