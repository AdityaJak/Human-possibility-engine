"""Persistence: namespaced SQLite, immutable event logs, checkpoints, stored-run loading.

Owned by the platform/backend lane (AGENTS.md).
"""

from hpe.storage.database import Database, now_iso
from hpe.storage.export import export_jsonl, export_parquet, parquet_available
from hpe.storage.migrations import MIGRATIONS, USER_PRIVATE_TABLES, Migration
from hpe.storage.repositories import (
    AuditLog,
    CheckpointRepository,
    EventRepository,
    EvidenceRepository,
    ExperimentRepository,
    ExternalDatasetRepository,
    RunRepository,
    TickRepository,
    UserRepository,
)
from hpe.storage.run_store import CheckpointStore, RunDirectory, StoredRun, StoredRunLoader
from hpe.storage.sinks import JsonlGzSink, SqliteSink, TeeSink, TickFrameIndex, sha256_of_gz_lines

__all__ = [
    "MIGRATIONS",
    "USER_PRIVATE_TABLES",
    "AuditLog",
    "CheckpointRepository",
    "CheckpointStore",
    "Database",
    "EventRepository",
    "EvidenceRepository",
    "ExperimentRepository",
    "ExternalDatasetRepository",
    "JsonlGzSink",
    "Migration",
    "RunDirectory",
    "RunRepository",
    "SqliteSink",
    "StoredRun",
    "StoredRunLoader",
    "TeeSink",
    "TickFrameIndex",
    "TickRepository",
    "UserRepository",
    "export_jsonl",
    "export_parquet",
    "now_iso",
    "parquet_available",
    "sha256_of_gz_lines",
]
