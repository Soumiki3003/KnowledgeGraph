from datetime import datetime

from pydantic import BaseModel, Field

from app import utils


class ChatUserMessageFormRequest(BaseModel):
    content: str = Field(description="Content of the message")
    created_at: datetime = Field(
        default_factory=utils.utc_now,
        description="Timestamp of when the message was created",
    )


class ChatResponse(BaseModel):
    answer: str
    hint_text: str | None = None
