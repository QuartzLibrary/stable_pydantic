import pydantic

class A_1(pydantic.BaseModel):
    other: str
    other_2: int

class A_0(pydantic.BaseModel):
    local: str
    local2: int

class Model_None(pydantic.BaseModel):
    local: A_0 = pydantic.Field(default_factory=lambda: A(local='test', local2=10))
    remote: A_1
    age: int
    ppp: int
