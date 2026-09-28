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
own lanes count up from 2. What a code *means* is not derived from the
class body — the run writes its lane names to
``run_metadata.rate_limit_lanes_json`` when it starts and hands them back
to :meth:`RateLimitTable.for_scraper` as ``stored_names`` on resume, so
the database is what says which lane code 2 is. Editing
``named_rate_limits`` between resumes therefore cannot regate pending rows
at a different rate: a reorder is ignored (the stored order wins), a new
lane is appended with a fresh code, and a lane the stored list names but
the scraper no longer declares is a :class:`ScraperConfigError` at run
start that names both lists. Zero for ``default`` is deliberately falsy: a
raw read of the column, outside :class:`RateLimitTable`, reads as
"default" the same way the old boolean ``bypass_rate_limit`` column read
as "not bypassed".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeGuard

from pyrate_limiter import Rate

from jkent.common.exceptions import ScraperConfigError

if TYPE_CHECKING:
    from jkent.common.scraper import BaseScraper

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


def _is_rate_list(rates: object) -> TypeGuard[Sequence[Rate]]:
    # A list or tuple of Rate objects. A bare Rate is truthy and passes an
    # emptiness check; ``list(rate)`` then fails far from the declaration.
    return isinstance(rates, (list, tuple)) and all(
        isinstance(r, Rate) for r in rates
    )


def _lane_order(
    named: Mapping[str, Sequence[Rate]],
    stored_names: Sequence[str] | None,
    *,
    owner: str,
) -> tuple[str, ...]:
    """The scraper's lanes in code order, after the reserved pair.

    Without *stored_names* that is declaration order. With them, the order
    the run already wrote wins, so a reorder of ``named_rate_limits``
    between resumes moves no row to another rate; lanes declared since are
    appended, taking fresh codes.
    """
    if stored_names is None:
        return tuple(named)
    reserved = len(RESERVED_RATE_LIMIT_NAMES)
    if tuple(stored_names[:reserved]) != RESERVED_RATE_LIMIT_NAMES:
        raise ScraperConfigError(
            f"{owner}: stored rate-limit lanes {list(stored_names)} do not "
            f"start with {list(RESERVED_RATE_LIMIT_NAMES)}; this is not a "
            "lane list jkent wrote"
        )
    stored = tuple(stored_names[reserved:])
    missing = [name for name in stored if name not in named]
    if missing:
        raise ScraperConfigError(
            f"{owner}.named_rate_limits no longer declares {missing}, which "
            f"this run stored as lanes (it recorded {list(stored_names)}, "
            f"the scraper now declares {list(named)}). Requests already "
            "queued on a dropped lane have nothing to gate them; restore "
            "the lane, or start a new run."
        )
    return stored + tuple(name for name in named if name not in stored)


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
    if len(rates) == 0:
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
        if len(rates) == 0:
            raise ScraperConfigError(
                f"{owner}.named_rate_limits[{name!r}]: a lane needs at "
                "least one Rate (use rate_limit='none' for no limit)"
            )


@dataclass(frozen=True)
class RateLimitTable:
    """One scraper's lanes, in code order, with the rates behind each.

    ``names[code]`` is the lane the database stores as ``code``. Built once
    per run by :meth:`for_scraper`; the queue uses it to encode and decode
    ``requests.rate_limit``, the run uses it to build one limiter per lane,
    and :attr:`names` is what a fresh run persists to
    ``run_metadata.rate_limit_lanes_json`` and hands back on resume.

    Attributes:
        names: Lane names indexed by code. ``names[0]`` is ``default`` and
            ``names[1]`` is ``none``; the rest follow the run's stored lane
            order, or the scraper's ``named_rate_limits`` in declaration
            order on a run that has not stored one yet.
        rates: The ``Rate`` objects behind each lane name. ``None`` for a
            lane with no limit: always ``none``, and ``default`` when the
            scraper declares no ``rate_limits``.
    """

    names: tuple[str, ...]
    rates: Mapping[str, Sequence[Rate] | None]

    @classmethod
    def for_scraper(
        cls,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any],
        *,
        stored_names: Sequence[str] | None = None,
    ) -> RateLimitTable:
        """The table for *scraper* (a ``BaseScraper`` subclass or instance).

        Args:
            scraper: The scraper whose ``rate_limits`` and
                ``named_rate_limits`` describe the lanes. Both are
                revalidated here — ``__init_subclass__`` has already checked
                a class written in a module, but a class built at runtime
                reaches this first.
            stored_names: The lane names this run recorded when it started
                (``run_metadata.rate_limit_lanes_json``), which fix the
                codes already written to ``requests.rate_limit``. Omit on a
                fresh run: the codes then follow declaration order, and the
                caller persists :attr:`names`.

        Raises:
            ScraperConfigError: from the validators, or when *stored_names*
                is not a lane list this scraper could have written (it must
                begin with the reserved pair), or names a lane the scraper
                no longer declares.
        """
        default_rates = scraper.rate_limits
        named = scraper.named_rate_limits
        cls_ = scraper if isinstance(scraper, type) else type(scraper)
        owner = cls_.__name__
        validate_rate_limits(default_rates, owner=owner)
        validate_named_rate_limits(named, owner=owner)
        names = RESERVED_RATE_LIMIT_NAMES + _lane_order(
            named, stored_names, owner=owner
        )
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
            ValueError: for a code past the end of the table — a row from a
                run whose stored lanes were not the ones this table was
                built with. Loud on purpose: silently gating it at the
                default rate would hide the mismatch.
        """
        if code == 0:
            return None
        if code < 0 or code >= len(self.names):
            raise ValueError(
                f"Stored rate limit code {code} has no lane; this table "
                f"holds {list(self.names)}. Was it built without the run's "
                "stored lane names?"
            )
        return self.names[code]
