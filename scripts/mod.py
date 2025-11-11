import pydantic


class A(pydantic.BaseModel):
    other: str
    other_2: int
