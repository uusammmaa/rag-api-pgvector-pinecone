"""Guards against drift between code and the files operators use."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.vectorstores.pgvector import schema_sql

ROOT = Path(__file__).resolve().parent.parent


def test_schema_file_matches_store_ddl() -> None:
    sql = (ROOT / "db" / "schema.sql").read_text()
    body = "\n".join(line for line in sql.splitlines() if not line.startswith("--")).strip()
    assert body == ";\n\n".join(schema_sql(1536)) + ";"


def test_env_example_only_lists_known_settings() -> None:
    known = {name.upper() for name in Settings.model_fields} | {"OPEN_AI_API_KEY"}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            assert line.split("=", 1)[0] in known, line


def test_env_example_has_no_real_secrets() -> None:
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "KEY=" in line and not line.startswith("#"):
            assert line.split("=", 1)[1].startswith("your-"), line
