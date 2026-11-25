"""Pydantic models for the backend's response contract.
Strict types and extra="forbid" make the models fail on a wrong type or an unexpected field."""
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr
from pydantic import UUID4


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UploadSuccess(_Strict):
    message: StrictStr
    session_id: UUID4


class Source(_Strict):
    source: StrictStr
    page: StrictInt


class AskSuccess(_Strict):
    answer: StrictStr
    answerable: StrictBool
    sources: list[Source]


class ErrorBody(_Strict):
    detail: StrictStr