"""Tests for the ``RequestStatus`` groups the query layer and stats rely on."""

from __future__ import annotations

from jkent.driver.database_engine.enums import RequestStatus


def test_status_groups_partition_request_status() -> None:
    """Active, terminal and parked cover every status exactly once.

    Read through the classmethods, so editing a group fails this.
    """
    groups = {
        "active": RequestStatus.active(),
        "terminal": RequestStatus.terminal(),
        "parked": RequestStatus.parked(),
    }

    for status in RequestStatus:
        hits = [name for name, group in groups.items() if status in group]
        assert len(hits) == 1, f"{status.name} is in {len(hits)} groups"

    assert RequestStatus.dequeuable() <= groups["active"]
