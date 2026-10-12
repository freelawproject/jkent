"""Tests for the ``RequestStatus`` groups the query layer and stats rely on."""

from __future__ import annotations

import pytest

from jkent.driver.database_engine.enums import RequestStatus

Groups = dict[str, frozenset[RequestStatus]]


def _group_problems(
    groups: Groups, dequeuable: frozenset[RequestStatus]
) -> list[str]:
    """Why ``groups`` fail to partition ``RequestStatus``; empty when they do.

    Every status must sit in exactly one group, and every dequeuable status
    must be active.
    """
    problems = [
        f"{status.name} is in {hits} groups"
        for status in RequestStatus
        if (hits := sum(status in group for group in groups.values())) != 1
    ]
    if not dequeuable <= groups["active"]:
        problems.append("a dequeuable status is not active")
    return problems


def _current_groups() -> Groups:
    return {
        "active": RequestStatus.active(),
        "terminal": RequestStatus.terminal(),
        "parked": RequestStatus.parked(),
    }


def test_status_groups_partition_request_status() -> None:
    """Active, terminal and parked cover every status exactly once."""
    assert _group_problems(_current_groups(), RequestStatus.dequeuable()) == []


@pytest.mark.parametrize(
    ("groups", "dequeuable", "expected"),
    [
        pytest.param(
            {
                "active": frozenset({RequestStatus.PENDING}),
                "terminal": frozenset(
                    {RequestStatus.COMPLETED, RequestStatus.FAILED}
                ),
                "parked": frozenset({RequestStatus.STUBBED}),
            },
            frozenset({RequestStatus.PENDING}),
            ["IN_PROGRESS is in 0 groups"],
            id="status-in-no-group",
        ),
        pytest.param(
            {
                "active": frozenset(
                    {RequestStatus.PENDING, RequestStatus.IN_PROGRESS}
                ),
                "terminal": frozenset(
                    {
                        RequestStatus.COMPLETED,
                        RequestStatus.FAILED,
                        RequestStatus.STUBBED,
                    }
                ),
                "parked": frozenset({RequestStatus.STUBBED}),
            },
            frozenset({RequestStatus.PENDING}),
            ["STUBBED is in 2 groups"],
            id="status-in-two-groups",
        ),
        pytest.param(
            {
                "active": frozenset(
                    {RequestStatus.PENDING, RequestStatus.IN_PROGRESS}
                ),
                "terminal": frozenset(
                    {RequestStatus.COMPLETED, RequestStatus.FAILED}
                ),
                "parked": frozenset({RequestStatus.STUBBED}),
            },
            frozenset({RequestStatus.STUBBED}),
            ["a dequeuable status is not active"],
            id="dequeuable-not-active",
        ),
    ],
)
def test_group_check_catches_inconsistent_groups(
    groups: Groups,
    dequeuable: frozenset[RequestStatus],
    expected: list[str],
) -> None:
    """The partition check reports each way the groups can go wrong."""
    assert _group_problems(groups, dequeuable) == expected


def _only_completed(cls: type[RequestStatus]) -> frozenset[RequestStatus]:
    return frozenset({RequestStatus.COMPLETED})


def test_partition_test_sees_the_live_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partition test reads the classmethods, so editing a group fails it."""
    monkeypatch.setattr(
        RequestStatus,
        "terminal",
        classmethod(_only_completed),
    )
    with pytest.raises(AssertionError, match="FAILED is in 0 groups"):
        test_status_groups_partition_request_status()
