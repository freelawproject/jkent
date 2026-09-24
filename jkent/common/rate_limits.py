"""Rate-limit lanes: the names a scraper declares and the codes a run stores.

A scraper's requests are gated by one of its *lanes*. Every scraper has two:
``default`` (whatever its ``rate_limits`` says — unlimited when that is
``None``) and ``none`` (never throttled, for fetches that do not hit the
scraped origin, such as expiring presigned download links). A scraper may
add more through ``named_rate_limits``, a mapping of lane name to
``pyrate_limiter.Rate`` objects::

    class MyScraper(BaseScraper[Case]):
        rate_limits = [Rate(2, Duration.SECOND)]
        named_rate_limits = {"downloads": [Rate(1, Duration.SECOND * 5)]}

Requests name their lane (``Request(rate_limit="downloads")``, or
``@step(rate_limit=...)`` on the target step, inherited the way priority
is); ``None`` means the default lane.

The run database stores the lane as a small integer in
``requests.rate_limit``: ``default`` is 0, ``none`` is 1, and the scraper's
own lanes count up from 2 in declaration order. The codes are derived from
order rather than spelled out because the stakes are low — the lane is a
courtesy rate, never a correctness invariant — so the one rule for scraper
authors is **append lanes, never insert or reorder them**: a run resumed
after a reorder would gate its pending rows at the wrong (but still
declared) rate. Zero for ``default`` is deliberately falsy: a raw read of
the column, outside :class:`RateLimitTable`, reads as "default" the same
way the old boolean ``bypass_rate_limit`` column read as "not bypassed".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pyrate_limiter import Rate

from jkent.common.exceptions import ScraperConfigError

__all__ = [
    "DEFAULT_RATE_LIMIT",
    "NO_RATE_LIMIT",
    "RESERVED_RATE_LIMIT_NAMES",
    "RateLimitTable",
    "validate_named_rate_limits",
    "validate_rate_limits",
]

#: The lane every request is in unless it says otherwise: the scraper's
#: ``rate_limits``. Stored as code 0.
DEFAULT_RATE_LIMIT: Final = "default"

#: The never-throttled lane. Stored as code 1.
NO_RATE_LIMIT: Final = "none"

#: Lane names the framework owns; a scraper's ``named_rate_limits`` may not
#: redeclare them.
RESERVED_RATE_LIMIT_NAMES: Final[tuple[str, ...]] = (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
)


def _is_rate_list(rates: object) -> bool:
    # A list or tuple of Rate objects. A bare Rate is truthy and passes an
    # emptiness check; ``list(rate)`` then fails far from the declaration.
    return isinstance(rates, (list, tuple)) and all(
        isinstance(r, Rate) for r in rates
    )


def validate_rate_limits(rates: object, *, owner: str) -> None:
    """Reject a ``rate_limits`` declaration that is not ``None`` or Rates.

    Checked where :func:`validate_named_rate_limits` is.

    Raises:
        ScraperConfigError: on anything but ``None`` or a non-empty list of
            ``Rate`` objects. ``[]`` is refused rather than read as "no
            limit" by truthiness; ``None`` says that.
    """
    if rates is None:
        return
    if not _is_rate_list(rates):
        raise ScraperConfigError(
            f"{owner}.rate_limits: expected None or a list of Rate, got "
            f"{rates!r}"
        )
    if not rates:
        raise ScraperConfigError(
            f"{owner}.rate_limits: an empty list; use None for no limit"
        )


def validate_named_rate_limits(
    named: Mapping[str, Sequence[Rate]], *, owner: str
) -> None:
    """Reject a ``named_rate_limits`` declaration that cannot become lanes.

    Checked at class definition (``BaseScraper.__init_subclass__``) so the
    error names the scraper, and again when a table is built from an object
    that is not a ``BaseScraper`` subclass.

    A ``Rate`` with a non-positive limit or interval cannot be constructed
    (``pyrate_limiter`` asserts on both), so only the shape is checked here.

    Raises:
        ScraperConfigError: on a reserved or non-string name, a value
            that is not a list of ``Rate`` objects, or an empty rate list.
    """
    for name, rates in named.items():
        if not isinstance(name, str) or not name:
            raise ScraperConfigError(
                f"{owner}.named_rate_limits: lane names must be non-empty "
                f"strings, got {name!r}"
            )
        if name in RESERVED_RATE_LIMIT_NAMES:
            raise ScraperConfigError(
                f"{owner}.named_rate_limits: {name!r} is reserved "
                f"(the framework provides {RESERVED_RATE_LIMIT_NAMES})"
            )
        if not _is_rate_list(rates):
            raise ScraperConfigError(
                f"{owner}.named_rate_limits[{name!r}]: expected a list of "
                f"Rate, got {rates!r}"
            )
        if not rates:
            raise ScraperConfigError(
                f"{owner}.named_rate_limits[{name!r}]: a lane needs at "
                "least one Rate (use rate_limit='none' for no limit)"
            )


@dataclass(frozen=True)
class RateLimitTable:
    """One scraper's lanes, in code order, with the rates behind each.

    ``names[code]`` is the lane the database stores as ``code``. Built once
    per run by :meth:`for_scraper`; the queue uses it to encode and decode
    ``requests.rate_limit`` and the run uses it to build one limiter per
    lane.

    Attributes:
        names: Lane names indexed by code. ``names[0]`` is ``default`` and
            ``names[1]`` is ``none``; the rest follow the scraper's
            ``named_rate_limits`` in declaration order.
        rates: The ``Rate`` objects behind each lane name. ``None`` for a
            lane with no limit: always ``none``, and ``default`` when the
            scraper declares no ``rate_limits``.
    """

    names: tuple[str, ...]
    rates: Mapping[str, Sequence[Rate] | None]

    @classmethod
    def for_scraper(cls, scraper: type[Any] | Any) -> RateLimitTable:
        """The table for *scraper* (a ``BaseScraper`` subclass or instance).

        Reads ``rate_limits`` and ``named_rate_limits`` through ``getattr``
        so a minimal stand-in without either attribute yields the bare
        two-lane table.
        """
        default_rates = getattr(scraper, "rate_limits", None)
        named: Mapping[str, Sequence[Rate]] = (
            getattr(scraper, "named_rate_limits", None) or {}
        )
        owner = getattr(scraper, "__name__", None) or type(scraper).__name__
        validate_rate_limits(default_rates, owner=owner)
        validate_named_rate_limits(named, owner=owner)
        names = RESERVED_RATE_LIMIT_NAMES + tuple(named)
        rates: dict[str, Sequence[Rate] | None] = {
            DEFAULT_RATE_LIMIT: list(default_rates) if default_rates else None,
            NO_RATE_LIMIT: None,
            **{name: list(r) for name, r in named.items()},
        }
        return cls(names=names, rates=rates)

    @classmethod
    def bare(cls) -> RateLimitTable:
        """The two framework lanes only, with an unlimited default."""
        return cls(
            names=RESERVED_RATE_LIMIT_NAMES,
            rates={DEFAULT_RATE_LIMIT: None, NO_RATE_LIMIT: None},
        )

    def encode(self, name: str | None) -> int:
        """The stored code for a ``Request.rate_limit`` value.

        ``None`` (the field's unset value) and ``"default"`` both encode to
        0.

        Raises:
            ValueError: for a lane this scraper does not declare — the
                failure lands at enqueue, on the step that yielded it.
        """
        if name is None:
            return 0
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown rate limit {name!r}; this scraper's lanes are "
                f"{list(self.names)}"
            ) from None

    def decode(self, code: int) -> str | None:
        """The ``Request.rate_limit`` value for a stored code.

        0 decodes to ``None``, the field's unset value, so a request
        round-trips through the database equal to the one enqueued.

        Raises:
            ValueError: for a code past the end of the table — the lane was
                removed from the scraper after this row was written. Loud on
                purpose: silently gating it at the default rate would hide
                the edit.
        """
        if code == 0:
            return None
        if code < 0 or code >= len(self.names):
            raise ValueError(
                f"Stored rate limit code {code} has no lane; this scraper "
                f"declares {list(self.names)}. Was a lane removed from "
                "named_rate_limits after this run was written?"
            )
        return self.names[code]
