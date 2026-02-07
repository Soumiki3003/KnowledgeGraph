from datetime import datetime, timezone
from pydantic import computed_field, model_validator
from sqlmodel import SQLModel, Field, Relationship, Index

from app import utils


class Student(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(min_length=2)
    email: str = Field(
        min_length=5,
        regex=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        index=True,
        unique=True,
    )
    password: str = Field(min_length=8)
    enabled: bool = Field(default=True, index=True)

    trajectory: list["StudentTrajectory"] = Relationship(
        back_populates="student",
        cascade_delete=True,
    )


class StudentTrajectory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str
    retrieved_nodes: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    interaction_type: str
    query_repeat_count: int = Field(0, ge=0)
    node_entry_count: int = Field(0, ge=0)
    response_time_sec: float = Field(0.0, ge=0.0)
    hint_triggered: bool = False
    hint_reason: str | None = None
    hint_text: str | None = None

    @computed_field
    @property
    def query_hash(self) -> str:
        return utils.hash_string(self.query)

    student_id: int = Field(foreign_key="student.id", ondelete="CASCADE")
    student: Student = Relationship(back_populates="trajectory")

    @model_validator(mode="after")
    def validate_scores_length(self):
        if len(self.scores) != len(self.retrieved_nodes):
            raise ValueError("Length of scores must match length of retrieved_nodes")
        return self

    @model_validator(mode="after")
    def validate_hint_fields(self):
        if self.hint_triggered:
            if not self.hint_reason:
                raise ValueError(
                    "hint_reason must be provided if hint_triggered is True"
                )
            if not self.hint_text:
                raise ValueError("hint_text must be provided if hint_triggered is True")
        return self

    __table_args__ = (Index("idx_query_hash", "query_hash"),)
