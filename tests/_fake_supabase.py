"""
A minimal in-memory stand-in for the Supabase client, shared by tests
that need real stateful CRUD sequences (insert then select then update,
etc.) rather than just recording which calls were made. Not a test
module itself - no test_ prefix, not collected by pytest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any


class _InMemoryTable:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._next_id = 1


class _FakeQueryBuilder:
    def __init__(self, table: _InMemoryTable):
        self._table = table
        self._filters: dict[str, Any] = {}
        self._op: str | None = None
        self._payload: dict[str, Any] | list[dict[str, Any]] | None = None
        self._order_key: str | None = None
        self._order_desc = False
        self._limit: int | None = None
        self._on_conflict: str | None = None

    def select(self, *_args: Any, **_kwargs: Any) -> "_FakeQueryBuilder":
        self._op = "select"
        return self

    def insert(self, payload: dict[str, Any] | list[dict[str, Any]]) -> "_FakeQueryBuilder":
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload: dict[str, Any] | list[dict[str, Any]], on_conflict: str = "id", **_kwargs: Any) -> "_FakeQueryBuilder":
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def update(self, payload: dict[str, Any]) -> "_FakeQueryBuilder":
        self._op = "update"
        self._payload = payload
        return self

    def delete(self) -> "_FakeQueryBuilder":
        self._op = "delete"
        return self

    def eq(self, key: str, value: Any) -> "_FakeQueryBuilder":
        self._filters[key] = value
        return self

    def order(self, key: str, desc: bool = False) -> "_FakeQueryBuilder":
        self._order_key = key
        self._order_desc = desc
        return self

    def limit(self, count: int) -> "_FakeQueryBuilder":
        self._limit = count
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        return all(row.get(key) == value for key, value in self._filters.items())

    def execute(self) -> SimpleNamespace:
        if self._op == "select":
            rows = [row for row in self._table.rows if self._matches(row)]
            if self._order_key:
                rows = sorted(rows, key=lambda r: r[self._order_key], reverse=self._order_desc)
            if self._limit is not None:
                rows = rows[: self._limit]
            return SimpleNamespace(data=[dict(row) for row in rows])

        if self._op == "insert":
            assert self._payload is not None
            # Supabase's insert() accepts either one row (a dict) or a
            # batch of rows (a list of dicts) - support both, the same
            # way record_ingestion_run() batch-inserts pipeline_metrics
            # rows in one call.
            payloads = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted_rows: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc).isoformat()
            for payload in payloads:
                row = dict(payload)
                row.setdefault("id", str(self._table._next_id))
                # Simulate Postgres's `default now()` for timestamp
                # columns the real schema always populates, so ordering
                # by created_at works the same way it would against a
                # real table.
                row.setdefault("created_at", now)
                row.setdefault("updated_at", now)
                self._table._next_id += 1
                self._table.rows.append(row)
                inserted_rows.append(dict(row))
            return SimpleNamespace(data=inserted_rows)

        if self._op == "upsert":
            assert self._payload is not None
            # Mimics Postgres's ON CONFLICT (<on_conflict columns>) DO
            # UPDATE: a row matching every conflict column gets updated
            # in place; otherwise a new row is inserted. Supports both a
            # single dict and a batch (list of dicts), same as insert().
            conflict_columns = [c.strip() for c in (self._on_conflict or "id").split(",")]
            payloads = self._payload if isinstance(self._payload, list) else [self._payload]
            now = datetime.now(timezone.utc).isoformat()
            result_rows: list[dict[str, Any]] = []
            for payload in payloads:
                existing = next(
                    (
                        row
                        for row in self._table.rows
                        if all(row.get(col) == payload.get(col) for col in conflict_columns)
                    ),
                    None,
                )
                if existing is not None:
                    existing.update(payload)
                    existing["updated_at"] = now
                    result_rows.append(dict(existing))
                else:
                    row = dict(payload)
                    row.setdefault("id", str(self._table._next_id))
                    row.setdefault("created_at", now)
                    row.setdefault("updated_at", now)
                    self._table._next_id += 1
                    self._table.rows.append(row)
                    result_rows.append(dict(row))
            return SimpleNamespace(data=result_rows)

        if self._op == "update":
            assert self._payload is not None
            matched = [row for row in self._table.rows if self._matches(row)]
            for row in matched:
                row.update(self._payload)
            return SimpleNamespace(data=[dict(row) for row in matched])

        if self._op == "delete":
            matched = [row for row in self._table.rows if self._matches(row)]
            self._table.rows = [row for row in self._table.rows if row not in matched]
            return SimpleNamespace(data=[dict(row) for row in matched])

        raise AssertionError("no operation (select/insert/upsert/update/delete) was called before execute()")


class FakeSupabaseClient:
    """In-memory stand-in for the Supabase client. Each table name gets its own independent row store."""

    def __init__(self) -> None:
        self._tables: dict[str, _InMemoryTable] = {}

    def table(self, name: str) -> _FakeQueryBuilder:
        self._tables.setdefault(name, _InMemoryTable())
        return _FakeQueryBuilder(self._tables[name])
