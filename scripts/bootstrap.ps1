$ErrorActionPreference = "Stop"

uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
uv run pytest
uv run ruff check .
