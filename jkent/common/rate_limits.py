"""Rate-limit lanes: the names a scraper declares and the codes a run stores.

A scraper's requests are gated by one of its *lanes*. Every scraper has two:
``default`` (whatever its ``rate_limits`` says — unlimited when that is
``None``) and ``none`` (never throttled, for fetches that do not hit the
scraped origin, such as expiring presigned download links). A scraper may
add more through ``named_rate_limits``, a mapping of lane name to pace::

    class MyScraper(BaseScraper[Case]):
        rate_limits = CappedPace(seconds=0.5)
        named_rate_limits = {"downloads": ReactivePace(seconds=5)}

A lane's pace is one of three kinds. :class:`CappedPace` spaces request
*starts*: the next one goes at least ``seconds`` after the previous one
started, and time the lane did not use earns no credit. :class:`ReactivePace`
runs the lane one request at a time and rests ``seconds`` after each one
*finishes*, so a slow server stretches the pace by itself.
:class:`AdaptivePace` is a capped pace that slows when the server pushes
back and climbs back, never past ``seconds``, once it has gone quiet.

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
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    ValidationError,
)

from jkent.common.exceptions import ScraperConfigError

if TYPE_CHECKING:
    from jkent.common.scraper import BaseScraper

__all__ = [
    "DEFAULT_RATE_LIMIT",
    "AdaptivePace",
    "NO_RATE_LIMIT",
    "RESERVED_RATE_LIMIT_NAMES",
    "CappedPace",
    "LaneDeclarations",
    "LaneName",
    "Pace",
    "RateLimitTable",
    "ReactivePace",
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


class _PaceModel(BaseModel):
    """Shared config: a pace is an immutable value with no stray fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class CappedPace(_PaceModel):
    """A minimum gap between request starts.

    The next request on the lane starts at least ``seconds`` after the
    previous one started. The lane keeps no credit for time it did not use:
    after a quiet spell requests still go out one per gap, never in a burst.
    How many are in flight at once is up to the worker pool.

    Attributes:
        kind: Discriminator for serialized paces.
        seconds: The gap, in seconds. Fractions are fine: 120 requests a
            second is ``CappedPace(seconds=1 / 120)``.
    """

    kind: Literal["capped"] = "capped"
    seconds: float = Field(gt=0)


class ReactivePace(_PaceModel):
    """One request at a time, with a rest after each.

    The next request on the lane starts ``seconds`` after the previous one
    *finished*, so the lane is serial whatever the worker count, and a slow
    response slows the lane by exactly as long as it took.

    Attributes:
        kind: Discriminator for serialized paces.
        seconds: The rest after each response, in seconds. Zero means one
            at a time with no rest.
    """

    kind: Literal["reactive"] = "reactive"
    seconds: float = Field(ge=0)


class AdaptivePace(_PaceModel):
    """A capped pace that slows down when the server pushes back.

    The lane starts as a :class:`CappedPace` of ``seconds``. A 429 or a
    ``Retry-After`` moves it to a slower gap, and a quiet spell moves it
    back one step at a time, never to a gap shorter than ``seconds``. The
    run logs where the lane settled when it ends.

    Attributes:
        kind: Discriminator for serialized paces.
        seconds: The fastest gap the lane may use, in seconds.
    """

    kind: Literal["adaptive"] = "adaptive"
    seconds: float = Field(gt=0)


#: How a lane may be paced.
Pace = CappedPace | ReactivePace | AdaptivePace


def _not_reserved(name: str) -> str:
    if name in RESERVED_RATE_LIMIT_NAMES:
        raise ValueError(
            f"{name!r} is reserved (the framework provides "
            f"{RESERVED_RATE_LIMIT_NAMES})"
        )
    return name


#: A pace as a scraper declares one: an instance, never a dict shaped like
#: one, which validation would coerce and leave uncoerced on the class.
_DeclaredPace = (
    InstanceOf[CappedPace]
    | InstanceOf[ReactivePace]
    | InstanceOf[AdaptivePace]
)

#: A lane name a scraper may declare: non-empty, and not one of the
#: framework's own.
LaneName = Annotated[str, Field(min_length=1), AfterValidator(_not_reserved)]


class LaneDeclarations(BaseModel):
    """A scraper's ``rate_limits`` and ``named_rate_limits``, type-checked.

    The two class attributes are plain ``ClassVar``s, which nothing checks
    at runtime; building this model from them is what holds a declaration
    to its annotations.

    Attributes:
        rate_limits: The ``default`` lane's pace, or ``None`` for no limit.
        named_rate_limits: The scraper's own lanes, in declaration order. A
            lane with no limit is not declared here; it is
            ``rate_limit="none"``.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    rate_limits: _DeclaredPace | None = None
    named_rate_limits: Mapping[LaneName, _DeclaredPace] = {}

    @classmethod
    def of(
        cls, scraper: type[BaseScraper[Any]] | BaseScraper[Any]
    ) -> LaneDeclarations:
        """The declarations of *scraper* (a subclass or an instance).

        Called at class definition (``BaseScraper.__init_subclass__``) so
        the error lands on the scraper, and again when a table is built,
        which is the first check for an object that is not a ``BaseScraper``
        subclass.

        Returns:
            The validated declarations.

        Raises:
            ScraperConfigError: If either attribute does not fit its type.
                The message names the scraper and the attribute.
        """
        cls_ = scraper if isinstance(scraper, type) else type(scraper)
        owner = cls_.__name__
        try:
            return cls(
                rate_limits=scraper.rate_limits,
                named_rate_limits=scraper.named_rate_limits,
            )
        except ValidationError as exc:
            # One line per attribute or lane: a value that fits neither
            # pace reports once per kind, at the same place.
            problems: dict[str, list[str]] = {}
            for error in exc.errors():
                field, *rest = error["loc"]
                where = f"{owner}.{field}"
                if field == "named_rate_limits" and rest:
                    where += f"[{rest[0]!r}]"
                problems.setdefault(where, []).append(error["msg"])
            raise ScraperConfigError(
                "\n".join(
                    f"{where}: {'; '.join(msgs)}"
                    for where, msgs in problems.items()
                )
            ) from None


def _lane_order(
    named: Mapping[str, Pace],
    stored_names: Sequence[str] | None,
    *,
    owner: str,
) -> tuple[str, ...]:
    """The scraper's lanes in code order, after the reserved pair.

    Without *stored_names* that is declaration order. With them, the order
    the run already wrote wins, so a reorder of ``named_rate_limits``
    between resumes moves no row to another rate; lanes declared since are
    appended, taking fresh codes.

    Args:
        named: The scraper's declared lanes, in declaration order.
        stored_names: The full lane list the run recorded, or ``None`` on
            a fresh run.
        owner: The scraper's name, for error messages.

    Returns:
        The scraper's lane names in code order, reserved pair excluded.

    Raises:
        ScraperConfigError: If *stored_names* is not a lane list jkent
            wrote, or names a lane the scraper no longer declares.
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


@dataclass(frozen=True)
class RateLimitTable:
    """One scraper's lanes, in code order, with the pace behind each.

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
        paces: The pace behind each lane name. ``None`` for a lane with no
            limit: always ``none``, and ``default`` when the scraper
            declares no ``rate_limits``.
    """

    names: tuple[str, ...]
    paces: Mapping[str, Pace | None]

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
                ``named_rate_limits`` describe the lanes, read through
                :class:`LaneDeclarations`.
            stored_names: The lane names this run recorded when it started
                (``run_metadata.rate_limit_lanes_json``), which fix the
                codes already written to ``requests.rate_limit``. Omit on a
                fresh run: the codes then follow declaration order, and the
                caller persists :attr:`names`.

        Returns:
            The table, with the reserved pair at codes 0 and 1 and the
            scraper's own lanes after them.

        :meth:`LaneDeclarations.of` and :func:`_lane_order` raise
        :class:`ScraperConfigError` for an invalid declaration, or for
        *stored_names* that is not a lane list this scraper could have
        written.
        """
        declared = LaneDeclarations.of(scraper)
        default_pace = declared.rate_limits
        named = declared.named_rate_limits
        cls_ = scraper if isinstance(scraper, type) else type(scraper)
        owner = cls_.__name__
        names = RESERVED_RATE_LIMIT_NAMES + _lane_order(
            named, stored_names, owner=owner
        )
        paces: dict[str, Pace | None] = {
            DEFAULT_RATE_LIMIT: default_pace,
            NO_RATE_LIMIT: None,
            **named,
        }
        return cls(names=names, paces=paces)

    @classmethod
    def bare(cls) -> RateLimitTable:
        """The two framework lanes only, with an unlimited default.

        Returns:
            A table that round-trips ``None`` and ``"none"`` and rejects
            every scraper-declared lane.
        """
        return cls(
            names=RESERVED_RATE_LIMIT_NAMES,
            paces={DEFAULT_RATE_LIMIT: None, NO_RATE_LIMIT: None},
        )

    def encode(self, name: str | None) -> int:
        """The stored code for a ``Request.rate_limit`` value.

        ``None`` (the field's unset value) and ``"default"`` both encode to
        0.

        Args:
            name: The lane a request asked for, or ``None``.

        Returns:
            The integer ``requests.rate_limit`` stores.

        Raises:
            ScraperConfigError: for a lane this scraper does not declare —
                the failure lands at enqueue, on the step that yielded it.
        """
        if name is None:
            return 0
        try:
            return self.names.index(name)
        except ValueError:
            raise ScraperConfigError(
                f"Unknown rate limit {name!r}; this scraper's lanes are "
                f"{list(self.names)}"
            ) from None

    def decode(self, code: int) -> str | None:
        """The ``Request.rate_limit`` value for a stored code.

        0 decodes to ``None``, the field's unset value, so a request
        round-trips through the database equal to the one enqueued.

        Args:
            code: The integer read off a ``requests`` row.

        Returns:
            The lane name, or ``None`` for the default lane.

        Raises:
            ScraperConfigError: for a code past the end of the table — a row
                from a run whose stored lanes were not the ones this table
                was built with. Loud on purpose: silently gating it at the
                default rate would hide the mismatch. :meth:`encode` refuses
                the same mismatch at enqueue, so a code reaching here with
                no lane means the table was built against the wrong run.
        """
        if code == 0:
            return None
        if code < 0 or code >= len(self.names):
            raise ScraperConfigError(
                f"Stored rate limit code {code} has no lane; this table "
                f"holds {list(self.names)}. Was it built without the run's "
                "stored lane names?"
            )
        return self.names[code]
