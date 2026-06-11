"""Shared SQLite persistence layer for jkent drivers.

This package holds the database layer for the unified driver:

- ``models`` / ``database`` -- ORM models, session factory, and
  engine/schema initialization (fresh databases are stamped at
  ``database.BASELINE_VERSION`` in ``schema_info``).
- ``sql_manager`` -- the :class:`SQLManager` query/persistence API.
- ``compression`` / ``stats`` -- stored-response compression and run analytics.

Consumers import the submodules directly (e.g.
``from jkent.driver.database_engine.sql_manager import SQLManager``).
"""
