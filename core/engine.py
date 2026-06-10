# Loads and validates workflow YAML files for the core workflow layer.
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class WorkflowStepConfig(BaseModel):
    """Describes one step entry from a workflow YAML file.

    This model is used by core.engine before core.executor asks core.registry
    to create the matching step implementation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "type")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Remove accidental surrounding spaces and reject blank values."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class WorkflowDefinition(BaseModel):
    """Describes the full workflow file after it has been loaded.

    The executor reads this object to know the workflow name, starting inputs,
    and ordered step list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    steps: list[WorkflowStepConfig] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        """Normalize the workflow name and reject blank names."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("workflow name must not be blank")
        return stripped

    @model_validator(mode="after")
    def _reject_duplicate_step_ids(self) -> "WorkflowDefinition":
        """Reject duplicate step ids so state writes are predictable."""

        seen: set[str] = set()
        duplicates: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                duplicates.add(step.id)
            seen.add(step.id)
        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate workflow step id(s): {duplicate_list}")
        return self


def load_workflow(path: str | Path) -> WorkflowDefinition:
    """Read a YAML workflow file and return a validated workflow object.

    This function connects the workflows/ YAML files to the core execution
    layer. It uses PyYAML to parse the file and Pydantic to check that the
    workflow has the fields the executor needs.
    """

    workflow_path = Path(path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"workflow file not found: {workflow_path}")

    try:
        # Parse only YAML data. No workflow code is executed while loading.
        with workflow_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid workflow YAML syntax in '{workflow_path}': {exc}") from exc

    try:
        return WorkflowDefinition.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid workflow YAML '{workflow_path}': {exc}") from exc
