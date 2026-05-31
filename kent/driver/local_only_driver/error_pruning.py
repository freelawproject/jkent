"""HATEOAS-aware parent-walk used by mode 3 (``desc-error-free``).

For each errored row in a source DB, walk ``parent_request_id`` upward
until the first ancestor with ``hateoas = True``. If none is found,
walk to the root (``parent_request_id IS NULL``). The chosen anchor is
the row that gets re-seeded in the output as a pending entry request.

Descendants of that anchor (the errored rows themselves and any
siblings on the way down) are *excluded from the source index* so that
when the replay scraper yields them, they fall through to the miss
policy (typically ``stub``) and end up as pending in the output.
"""

from __future__ import annotations

from dataclasses import dataclass

from kent.driver.local_only_driver.source_index import SourceIndex


@dataclass(frozen=True)
class PruningPlan:
    """Result of the mode-3 pre-pass.

    Attributes:
        anchors: For each source DB, the list of ``(request_id, depth)``
            anchor rows that need to be re-seeded. ``depth`` is the
            number of parent hops walked from the original errored row
            (0 if the errored row itself is the anchor).
        excluded_request_ids: For each source DB, the set of request_ids
            to exclude from the index (every node from each anchor down
            to its errored descendants).
    """

    anchors: dict[int, list[tuple[int, int]]]
    excluded_request_ids: dict[int, set[int]]


def compute_pruning_plan(index: SourceIndex) -> PruningPlan:
    """Identify hateoas-anchor ancestors of every errored row.

    Walks each source DB's errored rows via :meth:`SourceIndex.iter_errored_rows`
    and :meth:`SourceIndex.fetch_parent_chain`. The first ancestor in the
    chain whose ``hateoas`` is True is chosen as the anchor; if no row in
    the chain is True, the root is chosen. Every node from the anchor
    inclusive down to the errored row is added to
    ``excluded_request_ids`` so they will be naturally missed by the
    index lookup and re-fetched fresh.
    """
    anchors: dict[int, list[tuple[int, int]]] = {}
    excluded: dict[int, set[int]] = {}
    for db_idx in range(len(index.source_db_paths)):
        db_anchors: list[tuple[int, int]] = []
        db_excluded: set[int] = set()
        for errored_id, _parent_id in index.iter_errored_rows(db_idx):
            chain = index.fetch_parent_chain(db_idx, errored_id)
            if not chain:
                continue
            anchor_depth = _pick_anchor_depth(chain)
            anchor_id = chain[anchor_depth][0]
            db_anchors.append((anchor_id, anchor_depth))
            for rid, _hateoas in chain[: anchor_depth + 1]:
                db_excluded.add(rid)
        anchors[db_idx] = db_anchors
        excluded[db_idx] = db_excluded
    return PruningPlan(anchors=anchors, excluded_request_ids=excluded)


def _pick_anchor_depth(
    chain: list[tuple[int, bool | None]],
) -> int:
    """Return the index of the chosen anchor in ``chain``.

    ``chain`` starts at the errored row (depth 0) and ends at the root.
    The anchor is the *first* row whose ``hateoas`` is True. If none of
    the chain entries are True, the root (chain[-1]) is the anchor.
    """
    for depth, (_rid, hateoas) in enumerate(chain):
        if hateoas is True:
            return depth
    return len(chain) - 1
