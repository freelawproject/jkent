"""Shared SQLite persistence layer for jkent drivers.

This package holds the database layer for the unified driver:

- ``models`` / ``scoped_session`` / ``database`` -- ORM models, session
  factory, and engine/schema initialization.
- ``migrations`` -- versioned schema migrations.
- ``sql_manager`` -- the :class:`SQLManager` query/persistence API.
- ``compression`` / ``stats`` -- stored-response compression and run analytics.

Consumers import the submodules directly (e.g.
``from jkent.driver.database_engine.sql_manager import SQLManager``).
"""
