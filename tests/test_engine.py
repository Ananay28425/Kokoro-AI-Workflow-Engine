# Tests workflow YAML loading, registry wiring, execution, and persistence calls.
from __future__ import annotations

from time import perf_counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cli.commands import app
from core.engine import load_workflow
from core.executor import WorkflowExecutor
from core.registry import StepRegistry, build_default_registry


def test_load_workflow_from_yaml(tmp_path: Path) -> None:
    """Check that core.engine converts valid YAML into a workflow object."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: test_workflow
steps:
  - id: first
    type: read_file
    config:
      path: input.txt
""",
        encoding="utf-8",
    )

    workflow = load_workflow(workflow_path)

    assert workflow.name == "test_workflow"
    assert workflow.steps[0].id == "first"
    assert workflow.steps[0].type == "read_file"


def test_load_workflow_rejects_invalid_yaml(tmp_path: Path) -> None:
    """Check that missing required YAML fields produce a clear error."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("steps: []", encoding="utf-8")

    try:
        load_workflow(workflow_path)
    except ValueError as exc:
        assert "invalid workflow YAML" in str(exc)
    else:
        raise AssertionError("expected invalid workflow to raise ValueError")


def test_load_workflow_rejects_duplicate_step_ids(tmp_path: Path) -> None:
    """Check that duplicate step ids are blocked before execution."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: duplicate_steps
steps:
  - id: same
    type: capture
  - id: same
    type: capture
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate workflow step"):
        load_workflow(workflow_path)


def test_load_workflow_rejects_unsafe_yaml_tags(tmp_path: Path) -> None:
    """Check that unsafe YAML tags produce a normal validation error."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid workflow YAML syntax"):
        load_workflow(workflow_path)


class CaptureStep:
    """Fake step used to test core.executor without real AI or file work."""

    name = "capture"

    def __init__(self, value: str) -> None:
        """Store the value this fake step will write into workflow state."""

        self._value = value

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> str:
        """Pretend to run a real step by writing a fixed value to state."""

        state[config["output_key"]] = self._value
        return self._value


class FakeRepository:
    """Fake storage repository that records save calls in memory."""

    def __init__(self) -> None:
        """Create an empty list for captured repository calls."""

        self.calls: list[dict[str, Any]] = []

    def save_run(
        self,
        *,
        workflow_name: str,
        status: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: str | None = None,
    ) -> int:
        """Record the save request and return a predictable fake id."""

        self.calls.append(
            {
                "workflow_name": workflow_name,
                "status": status,
                "inputs": inputs,
                "outputs": outputs,
                "error": error,
            }
        )
        return 42


class FailingRepository:
    """Fake repository that always fails so executor error handling is tested."""

    def save_run(
        self,
        *,
        workflow_name: str,
        status: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: str | None = None,
    ) -> int:
        """Raise a predictable storage error."""

        raise RuntimeError("storage unavailable")


class RaisingStep:
    """Fake step that raises an error during execution."""

    name = "raising"

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> str:
        """Raise a predictable step error."""

        raise ValueError("bad step")


def test_executor_runs_registered_steps_and_persists_result(tmp_path: Path) -> None:
    """Check that core.executor calls registry steps and storage correctly."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: execution_test
steps:
  - id: first
    type: capture
    config:
      output_key: answer
""",
        encoding="utf-8",
    )
    registry = StepRegistry()
    registry.register("capture", lambda: CaptureStep("done"))
    repository = FakeRepository()

    result = WorkflowExecutor(registry, repository).execute(load_workflow(workflow_path), {"request": "run"})

    assert result.run_id == 42
    assert result.state["answer"] == "done"
    assert result.state["first"] == "done"
    assert repository.calls[0]["status"] == "succeeded"
    assert repository.calls[0]["outputs"]["answer"] == "done"


def test_executor_records_failed_run_with_step_context(tmp_path: Path) -> None:
    """Check that failed steps are persisted with useful error context."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: failure_test
steps:
  - id: explode
    type: raising
""",
        encoding="utf-8",
    )
    registry = StepRegistry()
    registry.register("raising", RaisingStep)
    repository = FakeRepository()

    with pytest.raises(RuntimeError, match="workflow step 'explode'"):
        WorkflowExecutor(registry, repository).execute(load_workflow(workflow_path))

    assert repository.calls[0]["status"] == "failed"
    assert "bad step" in repository.calls[0]["error"]


def test_executor_preserves_original_error_when_failure_persistence_fails(tmp_path: Path) -> None:
    """Check that a storage outage does not hide the real step failure."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: failure_persistence_test
steps:
  - id: explode
    type: raising
""",
        encoding="utf-8",
    )
    registry = StepRegistry()
    registry.register("raising", RaisingStep)

    with pytest.raises(RuntimeError, match="bad step"):
        WorkflowExecutor(registry, FailingRepository()).execute(load_workflow(workflow_path))


def test_step_registry_rejects_duplicate_registration() -> None:
    """Check that duplicate step registrations do not silently overwrite behavior."""

    registry = StepRegistry()
    registry.register("capture", lambda: CaptureStep("first"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register("capture", lambda: CaptureStep("second"))


def test_default_registry_is_lazy_and_has_expected_step_types() -> None:
    """Check that default registry construction does not load local AI runtimes."""

    registry = build_default_registry()

    assert registry.registered_types() == ("read_file", "speak", "summarize")


def test_executor_performance_smoke_for_many_simple_steps(tmp_path: Path) -> None:
    """Check that executing many lightweight steps remains fast enough for CLI use."""

    steps = "\n".join(
        f"  - id: step_{index}\n    type: capture\n    config:\n      output_key: value_{index}"
        for index in range(250)
    )
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(f"name: performance_test\nsteps:\n{steps}\n", encoding="utf-8")
    registry = StepRegistry()
    registry.register("capture", lambda: CaptureStep("done"))

    started = perf_counter()
    result = WorkflowExecutor(registry).execute(load_workflow(workflow_path))
    elapsed = perf_counter() - started

    assert result.state["step_249"] == "done"
    assert elapsed < 1.0


def test_cli_validate_success(tmp_path: Path) -> None:
    """Check the CLI validation command for a valid workflow."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("name: cli_test\nsteps:\n  - id: one\n    type: capture\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(workflow_path)])

    assert result.exit_code == 0
    assert "valid" in result.output


def test_cli_validate_prints_friendly_error(tmp_path: Path) -> None:
    """Check the CLI validation command returns a usable error message."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("steps: []", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", str(workflow_path)])

    assert result.exit_code == 1
    assert "error" in result.output


def test_cli_run_uses_registry_and_prints_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the CLI run command through the full loader/executor/output path."""

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
name: cli_run_test
steps:
  - id: first
    type: capture
    config:
      output_key: answer
""",
        encoding="utf-8",
    )
    registry = StepRegistry()
    registry.register("capture", lambda: CaptureStep("done"))
    monkeypatch.setattr("cli.commands.build_default_registry", lambda: registry)

    result = CliRunner().invoke(app, ["run", str(workflow_path)])

    assert result.exit_code == 0
    assert "cli_run_test" in result.output
    assert "done" in result.output
