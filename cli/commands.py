# Defines the Typer CLI that users run to validate and execute workflows.
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.engine import load_workflow
from core.executor import WorkflowExecutor
from core.registry import build_default_registry
from storage.database import PostgresDatabase
from storage.repository import PostgresWorkflowRepository

app = typer.Typer(help="Local Jarvis workflow runner.")
console = Console()
error_console = Console(stderr=True)


def _exit_with_error(exc: Exception) -> None:
    """Print a clean CLI error and stop with exit code 1."""

    error_console.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


@app.command()
def validate(workflow: Path) -> None:
    """Validate a workflow YAML file without running its steps."""

    try:
        definition = load_workflow(workflow)
    except Exception as exc:
        _exit_with_error(exc)
    else:
        console.print(f"[green]valid[/green] {definition.name} ({len(definition.steps)} steps)")


@app.command()
def run(workflow: Path, persist: bool = typer.Option(False, help="Persist run metadata to PostgreSQL.")) -> None:
    """Run a workflow from YAML and print the final state in the terminal."""

    try:
        repository = None
        if persist:
            # Persistence connects the CLI to storage.repository and PostgreSQL.
            repository = PostgresWorkflowRepository(PostgresDatabase())
            repository.initialize()

        # The CLI connects YAML loading, step factories, execution, and output.
        definition = load_workflow(workflow)
        executor = WorkflowExecutor(build_default_registry(), repository)
        result = executor.execute(definition)
    except Exception as exc:
        _exit_with_error(exc)

    table = Table(title=f"Workflow: {result.workflow_name}")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in result.state.items():
        table.add_row(str(key), str(value))
    console.print(table)


if __name__ == "__main__":
    # Allows this file to be run directly during local development.
    app()
