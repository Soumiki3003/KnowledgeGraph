from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from werkzeug.datastructures import FileStorage


class KnowledgeUploadRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    files: list[FileStorage] = Field(
        min_length=1, description="List of files to be uploaded"
    )
    html_link: str | None = Field(
        default=None,
        description="Optional HTML link to be associated with the uploaded files",
    )


class KnowledgeUploadResponse(BaseModel):
    uploaded: list[Path]
    html_link: str | None
    graph_generated: bool
