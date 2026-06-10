# Runs validated workflows by calling registered step implementations in order.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.engine import WorkflowDefinition
from core.interfaces import WorkflowRepository
from core.registry import StepRegistry
from core.state import WorkflowState


logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutionResult:
    """Holds the final result of a workflow run.

    The CLI reads this object to print the final state, and the storage layer
    may receive the same state when persistence is enabled.
    """

    workflow_name: str
    state: WorkflowState
    status: str = "succeeded"
    run_id: int | None = None


class WorkflowExecutor:
    """Coordinates workflow execution across the core, step, and storage layers.

    It does not know how each step works. It asks StepRegistry to build the
    correct step, calls that step, stores each result in WorkflowState, and
    optionally records the run through WorkflowRepository.
    """

    def __init__(self, registry: StepRegistry, repository: WorkflowRepository | None = None) -> None:
        """Store the registry and optional repository used during execution."""

        self._registry = registry
        self._repository = repository

    def execute(
        self,
        workflow: WorkflowDefinition,
        inputs: dict[str, Any] | None = None,
    ) -> WorkflowExecutionResult:
        """Run every step in a workflow from top to bottom.

        Runtime inputs are merged with inputs from the YAML file. Each step can
        read and write shared state, so later steps can use values produced by
        earlier steps.
        """

        state = WorkflowState(workflow.inputs | (inputs or {}))
        try:
            for step_config in workflow.steps:
                # The registry is the factory boundary between YAML step type
                # names and concrete classes in the steps/ package.
                step = self._registry.create(step_config.type)
                try:
                    result = step.run(state, step_config.config)
                except Exception as exc:
                    raise RuntimeError(
                        f"workflow step '{step_config.id}' of type '{step_config.type}' failed: {exc}"
                    ) from exc
                state[step_config.id] = result
            run_id = self._save(workflow.name, "succeeded", inputs or {}, dict(state))
            return WorkflowExecutionResult(workflow.name, state, run_id=run_id)
        except Exception as exc:
            try:
                self._save(workflow.name, "failed", inputs or {}, dict(state), str(exc))
            except Exception:
                logger.exception("failed to persist failed workflow run for %s", workflow.name)
            raise

    def _save(
        self,
        workflow_name: str,
        status: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: str | None = None,
    ) -> int | None:
        """Save workflow metadata if a repository was provided.

        The executor stays usable without PostgreSQL by treating persistence as
        optional. When enabled, this calls the storage layer.
        """

        if self._repository is None:
            return None
        return self._repository.save_run(
            workflow_name=workflow_name,
            status=status,
            inputs=inputs,
            outputs=outputs,
            error=error,
        )
