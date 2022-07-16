from pydantic import BaseModel
from typing import List, Dict, Tuple

from submodules.model.business_objects import information_source


class TextRequest(BaseModel):
    project_id: str
    information_source_id: str
    config: str
    text: str
    run_individually: bool
    label_names: List[str]


class ProjectRequest(BaseModel):
    project_id: str
    payload_id: str


class RecordRequest(BaseModel):
    config: str
    project_id: str
    record_id: str
    attribute_id: str
    label_names: List[str]


class SampleRecordsRequest(BaseModel):
    project_id: str
    information_source_id: str
    label_names: List[str]
