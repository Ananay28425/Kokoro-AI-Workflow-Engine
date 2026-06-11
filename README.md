# Kokoro AI Workflow Engine

A highly modular, local-first workflow execution engine built in Python. It supports loading workflows defined in YAML, reading local files, summarizing content using local Large Language Models (LLMs via `llama-cpp-python`), and synthesizing speech from text using local Text-to-Speech (TTS via `kokoro`). The engine features built-in verification, structured schemas, optional PostgreSQL execution tracking, and a rich, developer-friendly Command-Line Interface (CLI).

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Project Directory Layout](#project-directory-layout)
3. [Module-by-Module, Class-by-Class, & Function-by-Function Analysis](#module-by-module-class-by-class--function-by-function-analysis)
   - [Core Module (`core/`)](#1-core-module-core)
   - [Steps Module (`steps/`)](#2-steps-module-steps)
   - [AI Module (`ai/`)](#3-ai-module-ai)
   - [Storage Module (`storage/`)](#4-storage-module-storage)
   - [CLI Module (`cli/`)](#5-cli-module-cli)
   - [Entry Point (`main.py`)](#6-entry-point-mainpy)
   - [Project Specification (`pyproject.toml`)](#7-project-specification-pyprojecttoml)
4. [Workflow Configuration Schema](#workflow-configuration-schema)
5. [Local Inference & Audio Setup](#local-inference--audio-setup)
6. [PostgreSQL Persistence Configuration](#postgresql-persistence-configuration)
7. [Running the CLI & Workflows](#running-the-cli--workflows)
8. [Developer Guide & Testing](#developer-guide--testing)

---

## System Architecture Overview

The Kokoro AI Workflow Engine is built on three core design principles:

1. **Strict Interface Boundaries (PEP 544 Protocols):** The execution engine, steps, AI components, and persistence layer are completely decoupled using Python `Protocol` interfaces. This allows effortless mocking and makes testing incredibly fast and reliable.
2. **Lazy Initialization of Heavy Runtimes:** Heavy libraries like `llama_cpp` and `kokoro` (along with their neural weights) are only loaded when an execution step actually requires them. Validating files, running tests, or displaying help messages takes milliseconds.
3. **Pydantic-Driven Configuration Safety:** All configuration boundaries (YAML files, command parameters, API requests) are strictly typed and parsed with `Pydantic`. Any configuration typo or invalid parameter is blocked _before_ runtime execution begins.

The diagram below shows how state and control flow through the system during a run:

```mermaid
flowchart TD
    CLI[cli/commands.py] -->|1. Parse & Validate| Engine[core/engine.py]
    CLI -->|2. Create Executor| Executor[core/executor.py]

    Engine -->|Loads YAML| Definition[WorkflowDefinition]
    Executor -->|3. Fetch Steps| Registry[core/registry.py]

    Registry -->|Resolves Types| ReadStep[steps/read_file.py]
    Registry -->|Resolves Types| SumStep[steps/summarize.py]
    Registry -->|Resolves Types| SpeakStep[steps/speak.py]

    Executor -->|4. Run Step Sequence| ReadStep
    Executor -->|5. Run Step Sequence| SumStep
    Executor -->|6. Run Step Sequence| SpeakStep

    SumStep -->|Inference request| LLMAdapter[ai/llama_cpp.py]
    SpeakStep -->|Synthesis request| TTSAdapter[ai/kokoro.py]

    Executor -.->|7. Optional Save| Repo[storage/repository.py]
    Repo -.->|Persist Run Data| DB[(PostgreSQL)]
```

---

## Project Directory Layout

The repository is structured as follows:

```text
kokoro-ai-workflow-engine/
├── pyproject.toml              # Project dependency setup, build system, and scripts
├── README.md                   # This detailed architectural guide and documentation
├── main.py                     # CLI entry-point script
│
├── workflows/                  # Reusable local YAML workflow definitions
│   ├── summarize.yaml          # Reads a file and summarizes it
│   ├── analyze_logs.yaml       # Reserved for automated log-analysis DAGs
│   └── speak_summary.yaml      # Reads, summarizes, and generates voice output
│
├── core/                       # Core execution and workflow engine mechanics
│   ├── engine.py               # YAML safely loader and model schema parser
│   ├── executor.py             # Orchestrator running steps in sequence
│   ├── registry.py             # Lazy loading step registry
│   ├── state.py                # Inter-step workflow run memory state
│   └── interfaces.py           # Shared interface definitions (Protocols)
│
├── steps/                      # Decoupled workflow action implementations
│   ├── base.py                 # Abstract base class for all workflow steps
│   ├── read_file.py            # Local UTF-8 text file reading step
│   ├── summarize.py            # Local model summarization step
│   ├── analyze_logs.py         # Placeholder for future log-analyzing steps
│   ├── generate_report.py      # Placeholder for future reporting steps
│   └── speak.py                # Local text-to-speech WAV generator step
│
├── ai/                         # Local AI model clients and adapters
│   ├── llm.py                  # Module contract helper for text-generation clients
│   ├── llama_cpp.py            # Client adapter for llama.cpp GGUF files
│   ├── kokoro.py               # Client adapter for Kokoro-82M TTS pipelines
│   ├── prompts.py              # Central prompts repository
│   └── schemas.py              # Pydantic schema validation for model queries
│
├── storage/                    # Database, schema migration, and data models
│   ├── database.py             # Context-managed connection pool wrapper for PostgreSQL
│   ├── models.py               # Read-only memory models for workflow tracking
│   ├── repository.py           # SQL repository layer writing workflow metadata
│   ├── hash_utils.py           # Hashing helper functions
│   └── migrations/             # Storage for SQL schema migration scripts
│
├── cli/                        # Rich user interfaces
│   └── commands.py             # Commands for validation, execution, and tables
│
├── assets/                     # Output storage
│   └── audio/                  # Target destination folder for speech output
│
├── examples/                   # Developer examples and samples
│
└── tests/                      # Core test suites (Fast unit-testing)
    ├── test_engine.py          # Validates core parser and CLI logic
    ├── test_steps.py           # Validates workflow step behaviors (uses fakes)
    ├── test_cache.py           # Validates DB connection logic and stable hashes
    └── test_speech.py          # Validates AI adapters using mock libraries
```

---

## Module-by-Module, Class-by-Class, & Function-by-Function Analysis

### 1. Core Module (`core/`)

The `core/` module is the backbone of the engine. It parses declarative YAML configuration files, manages the state of active runs, provides interfaces, and executes steps in a predictable sequence.

---

#### File: `core/interfaces.py`

This file houses PEP 544 structural subtyping `Protocol` models that act as the interface boundaries for key systems. Code in `core/` relies strictly on these protocols rather than concrete classes.

- `class Step(Protocol)`: Defines an executable workflow step.
  - `name: str`: Public label used for registration and log trace displays.
  - `def run(self, state: dict[str, Any], config: dict[str, Any]) -> Any`: Executed by the engine. Accepts a shared state dictionary and a local YAML configuration dictionary. Returns a value representing the step's primary outcome.
- `class LLMClient(Protocol)`: Interface for interacting with local language models.
  - `def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str`: Accepts a text prompt and configuration, then returns the generated text.
- `class TTSClient(Protocol)`: Interface for text-to-speech synthesizers.
  - `def synthesize(self, text: str, output_path: Path, *, voice: str | None = None) -> Path`: Synthesizes voice audio from text, writes it to a file at `output_path`, and returns the absolute or relative path to the completed file.
- `class WorkflowRepository(Protocol)`: Persistence engine contract.
  - `def save_run(self, *, workflow_name: str, status: str, inputs: dict[str, Any], outputs: dict[str, Any], error: str | None = None) -> int`: Persists state metadata from a run and returns the assigned database ID.

---

#### File: `core/state.py`

Provides state management mechanics used by steps to read and write variables during execution.

- `class WorkflowState(dict[str, Any])`: A mutable container inheriting from `dict` that holds global workflow variables.
  - `def require(self, key: str) -> Any`: Safely retrieves a variable from state. If the key is not present, raises a clear, user-friendly `KeyError` indicating that the workflow is missing a required input or step output.

---

#### File: `core/engine.py`

Contains the logic for reading YAML workflow definition files, verifying structure, and transforming raw YAML text into strong, typed configurations.

- `class WorkflowStepConfig(BaseModel)`: A strict Pydantic model representing a single step's configuration in a YAML file.
  - `id: str`: The unique runtime ID of the step (must be at least 1 character).
  - `type: str`: The YAML type of the step (e.g., `read_file`, `summarize`, `speak`).
  - `config: dict[str, Any]`: Arguments passed to the step when run.
  - `_strip_required_text(cls, value: str) -> str`: A Pydantic field validator that automatically strips whitespace and raises a `ValueError` for blank string IDs or types.
- `class WorkflowDefinition(BaseModel)`: Represents the entire workflow file schema.
  - `name: str`: Unique workflow identifier label.
  - `description: str | None`: Optional developer overview.
  - `inputs: dict[str, Any]`: Key-value map of default parameters.
  - `steps: list[WorkflowStepConfig]`: Sequential array of steps to run.
  - `_strip_name(cls, value: str) -> str`: Normalizes the name by removing accidental leading/trailing spaces and rejecting empty strings.
  - `_reject_duplicate_step_ids(self) -> "WorkflowDefinition"`: A model-validator that checks for ID collisions among steps. This prevents steps from accidentally overwriting other steps' output values in `WorkflowState`.
- `def load_workflow(path: str | Path) -> WorkflowDefinition`: Reads a YAML file from disk, parses it using PyYAML's `safe_load`, and converts the raw dictionary into a validated `WorkflowDefinition` object. Raises appropriate `FileNotFoundError`, YAML syntax error, or validation error.

---

#### File: `core/registry.py`

Manages step type mappings and decouples step construction from execution.

- `class StepRegistry`: A central factory registry.
  - `register(self, step_type: str, factory: StepFactory) -> None`: Registers a step type mapping (e.g., `"read_file"`) with a factory function. Raises a `ValueError` if the step type is blank or already registered.
  - `create(self, step_type: str) -> Step`: Executes the factory function associated with `step_type` and returns an active `Step` object. Throws a helpful `KeyError` if the step type is unknown.
  - `registered_types(self) -> tuple[str, ...]`: Returns an alphabetically sorted tuple of all registered step types.
- `def build_default_registry() -> StepRegistry`: Assembly helper used by the production CLI. It registers default steps: `"read_file"` to `ReadFileStep`, `"summarize"` to a factory invoking `LlamaCppAdapter`, and `"speak"` to a factory invoking `KokoroTTSAdapter`. Heavy import statements are placed inside this function to keep CLI imports fast.

---

#### File: `core/executor.py`

Executes workflows from top to bottom, updating state variables and persisting run logs.

- `class WorkflowExecutionResult`:
  - `workflow_name: str`, `state: WorkflowState`, `status: str`, `run_id: int | None`.
- `class WorkflowExecutor`:
  - `def __init__(self, registry: StepRegistry, repository: WorkflowRepository | None = None)`: Initializes the executor with step construction factories and an optional persistence database repository.
  - `def execute(self, workflow: WorkflowDefinition, inputs: dict[str, Any] | None = None) -> WorkflowExecutionResult`: Merges runtime override inputs with the workflow's defaults. Executes steps sequentially:
    1.  Retrieves a step implementation from the registry using `step_config.type`.
    2.  Executes `step.run(state, step_config.config)`.
    3.  Stores the step's return value in the shared state under `state[step_config.id]`.
    4.  Saves the metadata through the repository if database persistence is enabled.
    5.  If a step fails, the executor catches the exception, updates the execution status to `"failed"`, logs the error to the database, and re-raises a detailed `RuntimeError` with step-specific debugging context.
  - `def _save(self, workflow_name, status, inputs, outputs, error) -> int | None`: Interacts with the repository to save execution run details. Suppresses database errors during failure cleanup to preserve the original step failure exception.

---

### 2. Steps Module (`steps/`)

The `steps/` directory contains individual task implementations. Each class inherits from a base model and conforms to the `Step` protocol.

---

#### File: `steps/base.py`

- `class BaseStep`: A lightweight abstract dataclass.
  - `name: str`: Stores the debugging name.
  - `def run(self, state, config)`: Abstract method that must be implemented by subclasses.

---

#### File: `steps/read_file.py`

- `class ReadFileConfig(BaseModel)`: Configuration schema for the `read_file` step.
  - `path: Path`: The target text file to read.
  - `output_key: str`: The state key where text content will be stored (defaults to `"text"`).
- `class ReadFileStep(BaseStep)`: Reads a file from disk.
  - `def run(self, state, config) -> str`: Validates configuration using `ReadFileConfig`. Ensures the file exists, reads its contents using UTF-8 encoding, writes the string to state under `output_key`, and returns the text. Raises `ValueError` for binary/invalid files.

---

#### File: `steps/summarize.py`

- `class SummarizeConfig(BaseModel)`: Validation schema for the LLM summarizer step.
  - `input_key: str`: The state key to read the source text from (defaults to `"text"`).
  - `output_key: str`: The state key where the summary is stored (defaults to `"summary"`).
  - `prompt: str`: Prompt template sent to the model (defaults to `"Summarize the following text clearly and concisely:\n\n{text}"`).
  - `max_tokens: int`: Maximum response length limit (1 to 8192, defaults to 512).
  - `temperature: float`: Creativity setting (0.0 to 2.0, defaults to 0.2).
  - `_validate_prompt_contract(cls, value: str) -> str`: Validates the prompt template. Ensures it contains `{text}` and no other placeholders.
- `class SummarizeStep(BaseStep)`: Coordinates prompt generation.
  - `def __init__(self, llm: LLMClient)`: Receives a client implementation conforming to `LLMClient`.
  - `def run(self, state, config) -> str`: Validates configuration. Reads source text from state, injects it into the prompt template using `.format(text=text)`, calls the local LLM model via the adapter, stores the result under `output_key`, and returns the summary text.

---

#### File: `steps/speak.py`

- `class SpeakConfig(BaseModel)`: Validation schema for speech generation.
  - `input_key: str`: State key containing the text to speak (defaults to `"summary"`).
  - `output_key: str`: State key where the file path is stored (defaults to `"audio_path"`).
  - `output_path: Path`: Destination path for the audio file (defaults to `assets/audio/output.wav`).
  - `voice: str | None`: Optional Kokoro voice code (e.g., `af_bella`, `am_adam`).
- `class SpeakStep(BaseStep)`: Invokes the local text-to-speech engine.
  - `def __init__(self, tts: TTSClient)`: Receives a client implementation conforming to `TTSClient`.
  - `def run(self, state, config) -> str`: Validates configuration. Reads text from state, triggers speech synthesis via the adapter, stores the absolute or relative file path under `output_key`, and returns the path string.

---

#### Files: `steps/analyze_logs.py` & `steps/generate_report.py`

These files are intentionally left as reserved placeholders, pre-wired for future step types.

---

### 3. AI Module (`ai/`)

The `ai/` module contains adapters for local neural network inference, isolating third-party runtime models behind the project's interfaces.

---

#### File: `ai/llama_cpp.py`

- `class LlamaCppAdapter`: Connects workflow steps to local LLMs via `llama-cpp-python`.
  - `def __init__(self, model_path: str | None = None, client: Any | None = None, **kwargs)`: Initializes the adapter. Accepts an optional custom `model_path` (defaults to the `LLAMA_CPP_MODEL_PATH` environment variable) or a pre-configured mock `client` for unit testing.
  - `def complete(self, prompt, *, max_tokens=512, temperature=0.2) -> str`: Submits a prompt to the model. Auto-detects and supports both standard method calls (`client()`) and older APIs (`client.create_completion()`).
  - `_get_client(self) -> Any`: Lazily instantiates the underlying `Llama` model from llama.cpp. This ensures the engine imports quickly and works for basic tasks even when no GGUF file is downloaded.
  - `_extract_text(self, response: Any) -> str`: Parses the response from llama.cpp. Handles differences in response shapes (e.g., `choices[0].text` vs. `choices[0].message.content`).

---

#### File: `ai/kokoro.py`

- `class KokoroTTSAdapter`: Adapts the Kokoro text-to-speech library to generate wave files locally.
  - `def __init__(self, model: str | None = None, *, lang_code: str = "a", sample_rate: int = 24000, pipeline: Any | None = None, **kwargs)`: Initializes the adapter with language options and lazy initialization configurations.
  - `def synthesize(self, text, output_path, *, voice=None) -> Path`: Synthesizes text into speech. Creates the parent directories if they don't exist, extracts generated audio data, writes it to disk, and returns the path to the completed file.
  - `_get_pipeline(self) -> Any`: Lazily imports and instantiates the `KPipeline` object. Falls back gracefully if custom model weights are not provided.
  - `_extract_audio_chunks(self, audio: Any) -> list[Any]`: Normalizes output from the Kokoro pipeline. Handles differences in formats, supporting raw sound bytes, tuple sequences, and custom generator chunks.
  - `_write_audio_output(self, audio: Any, output_path: Path) -> None`: Standardizes audio file generation. Writes raw bytes directly to disk, or processes numeric audio arrays using `numpy` and `soundfile` with the configured sample rate.

---

#### Files: `ai/prompts.py` & `ai/schemas.py` & `ai/llm.py`

- `ai/prompts.py`: Defines the reusable prompt template `SUMMARIZE_PROMPT`.
- `ai/schemas.py`: Defines validating Pydantic schemas: `CompletionRequest` (wrapping prompt settings) and `SpeechRequest` (wrapping text-to-speech voice selections).
- `ai/llm.py`: Cleanly re-exports the `LLMClient` protocol for structural imports.

---

### 4. Storage Module (`storage/`)

This module manages workflow history and stores execution metadata in a PostgreSQL database using the `psycopg` database driver.

---

#### File: `storage/database.py`

- `class PostgresDatabase`: Manages database connections.
  - `def __init__(self, dsn: str | None = None)`: Initializes the database manager. Fetches the connection string from the `DATABASE_URL` environment variable if a custom DSN is not provided.
  - `connect(self) -> Iterator[Any]`: A context manager that opens a connection, yields it, and automatically commits transactions or rolls them back if errors occur. Raises a clear error if `psycopg` is missing.

---

#### File: `storage/repository.py`

- `class PostgresWorkflowRepository`: Handles table creation and database interactions.
  - `def initialize(self) -> None`: Creates the `workflow_runs` table and indexes if they do not exist.
  - `def save_run(self, *, workflow_name, status, inputs, outputs, error=None) -> int`: Inserts a workflow run record into the database. Serializes input and output dictionaries to Pydantic-compatible JSON strings and returns the generated sequence ID.

---

#### Files: `storage/models.py` & `storage/hash_utils.py`

- `storage/models.py`: Declares `WorkflowRun`, a frozen data model representing stored database runs.
- `storage/hash_utils.py`: Declares `stable_sha256(value: str) -> str`, which computes deterministic, stable SHA-256 hashes for use in cache-key systems.

---

### 5. CLI Module (`cli/`)

The command-line interface provides tools to inspect, validate, and execute workflows locally.

---

#### File: `cli/commands.py`

- `app = typer.Typer(help="Local Jimmy workflow runner.")`: Creates the Typer CLI application.
- `def validate(workflow: Path) -> None`: Loads and validates a workflow YAML file without running its steps. Prints validation status and step counts using `Rich` styling.
- `def run(workflow: Path, persist: bool) -> None`: Runs a workflow from end to end. If `--persist` is specified, it initializes the database connection, registers the repository, executes the steps, and displays a summary table of the final state variables in the terminal.

---

### 6. Entry Point (`main.py`)

This script acts as the main execution entry point. When run from the command line, it forwards arguments directly to the Typer app inside the CLI module.

```python
from cli.commands import app

if __name__ == "__main__":
    app()
```

---

### 7. Project Specification (`pyproject.toml`)

Manages dependencies, script entry points, and optional feature flags.

- `[project]`: Defines metadata (such as version `0.1.0`) and core runtime dependencies (`pydantic`, `pyyaml`, `typer`, `rich`, `psycopg`).
- `[project.optional-dependencies]`:
  - `dev`: Installs `pytest` for running test suites.
  - `llm`: Installs `llama-cpp-python` for local AI text generation.
  - `tts`: Installs `kokoro`, `numpy`, and `soundfile` for local speech synthesis.
  - `local`: Installs both `llm` and `tts` dependencies for full offline execution.
- `[project.scripts]`: Maps the command-line shortcut `jarvis` (or `jimmy`) directly to `cli.commands:app`.

---

## Workflow Configuration Schema

Workflows are defined in simple YAML files. Here is the validated schema:

| Field         | Type          | Required | Description                                      |
| :------------ | :------------ | :------- | :----------------------------------------------- |
| `name`        | String        | **Yes**  | A clean, non-blank unique workflow identifier.   |
| `description` | String        | No       | A summary of what the workflow does.             |
| `inputs`      | Dictionary    | No       | Key-value pairs providing default parameters.    |
| `steps`       | List of steps | **Yes**  | Ordered array of step configurations to execute. |

### Individual Step Configuration (`steps`)

- `id`: String (**Required**). Unique step identifier used as the state output key.
- `type`: String (**Required**). The step type (e.g., `read_file`, `summarize`, `speak`).
- `config`: Dictionary (Optional). Step-specific parameters.

### Example Workflow (`workflows/speak_summary.yaml`)

```yaml
name: speak_summary
description: "Reads a local file, generates a summary, and converts that summary to audio."
inputs:
  input_file: "examples/input.txt"
  audio_output: "assets/audio/output.wav"
steps:
  - id: file_content
    type: read_file
    config:
      path: "{input_file}"
      output_key: "text"

  - id: summary_text
    type: summarize
    config:
      input_key: "text"
      output_key: "summary"
      prompt: "Synthesize the core message of this text into a single sentence:\n\n{text}"
      max_tokens: 256
      temperature: 0.3

  - id: speech_result
    type: speak
    config:
      input_key: "summary"
      output_key: "audio_path"
      output_path: "{audio_output}"
      voice: "af_bella"
```

---

## Local Inference & Audio Setup

To run local language models and speech synthesis without sending data to external APIs, follow these steps:

### 1. Download a GGUF Model

Download a GGUF model (such as Llama-3 or Mistral-7B) from Hugging Face, place it in a local directory, and point the environment variable to it:

```bash
# On Linux/macOS
export LLAMA_CPP_MODEL_PATH="/path/to/models/llama-3-8b-instruct.Q4_K_M.gguf"

# On Windows (PowerShell)
$env:LLAMA_CPP_MODEL_PATH="C:\path\to\models\llama-3-8b-instruct.Q4_K_M.gguf"
```

### 2. Set Up Voice Models for Kokoro

Kokoro downloads voice model weights automatically on its first run. To configure a specific voice or directory, consult the [Kokoro library documentation](https://github.com/hexgrad/kokoro).

---

## PostgreSQL Persistence Configuration

To save workflow metadata and run records in a database, configure your PostgreSQL connection:

1. Ensure a PostgreSQL instance is running.
2. Set the `DATABASE_URL` environment variable:
   ```bash
   export DATABASE_URL="postgresql://username:password@localhost:5432/workflow_db"
   ```
3. Run the workflow with the `--persist` flag. This will automatically initialize the `workflow_runs` table and record execution outcomes.

---

## Running the CLI & Workflows

Ensure dependencies are installed:

```bash
# Install core dependencies along with local AI runtimes
pip install -e ".[local]"
```

### 1. Validate a Workflow

Syntax-check and validate a workflow file without running any code:

```bash
python main.py validate workflows/speak_summary.yaml
```

### 2. Execute a Workflow

Run a workflow and view its execution trace:

```bash
python main.py run workflows/speak_summary.yaml
```

To run the workflow and save its metadata to PostgreSQL:

```bash
python main.py run workflows/speak_summary.yaml --persist
```

---

## Developer Guide & Testing

The project uses `pytest` for testing. The test suite uses mock classes and simulated pipelines, so you can run tests quickly without needing GGUF models or GPU acceleration.

### Run All Tests

```bash
pytest
```

### Core Tests Overview

- `tests/test_steps.py`: Verifies that file reading, LLM prompts, and TTS voice steps validate configurations correctly and update state variables as expected.
- `tests/test_cache.py`: Verifies SHA-256 hash consistency and database repository operations.
- `tests/test_engine.py`: Verifies YAML loading, validation logic, and Typer CLI actions.
- `tests/test_speech.py`: Verifies llama.cpp response extraction and Kokoro speech output generation using simulated clients.
