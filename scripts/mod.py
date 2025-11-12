import pydantic


class A(pydantic.BaseModel):
    remote: str
    remote_2: int
