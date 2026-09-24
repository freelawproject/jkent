"""Incidental request storage operations with content deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import delete, select

from jkent.driver.database_engine.models import (
    IncidentalRequest,
    IncidentalRequestStorage,
)
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import (
    IncidentalCapture,
    IncidentalRequestRecord,
    model_columns,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select

#: Which :class:`IncidentalCapture` fields land on which table, by column
#: name. The split is the schema's: everything particular to one fetch is a
#: column of ``incidental_requests``, and only the payload (plus its
#: representative response headers) is a column of the content-addressed
#: storage table, so the two sets are disjoint.
_STORAGE_FIELDS: Final = set(
    model_columns(IncidentalCapture, IncidentalRequestStorage, partial=True)
)
_REQUEST_FIELDS: Final = set(
    model_columns(IncidentalCapture, IncidentalRequest, partial=True)
)
#: :class:`IncidentalRequestRecord`'s columns: ``incidental_requests`` first,
#: the storage row for the rest.
_RECORD_COLUMNS: Final = model_columns(
    IncidentalRequestRecord, IncidentalRequest, IncidentalRequestStorage
).values()


def incidental_record_select() -> Select[Any]:
    """The select() an :class:`IncidentalRequestRecord` is validated from."""
    return select(*_RECORD_COLUMNS).outerjoin(
        IncidentalRequestStorage,
        IncidentalRequest.storage_id == IncidentalRequestStorage.id,
    )


class IncidentalRequestStorageMixin(SQLManagerBase):
    """Insert and retrieve incidental browser requests with content dedup."""

    async def replace_incidental_requests(
        self,
        parent_request_id: int,
        records: list[IncidentalCapture],
    ) -> list[int]:
        """Store a navigation's captured incidentals in one transaction.

        The batch *replaces* whatever the parent already had. A retried
        navigation captures its sub-requests again, and the request row
        describes its latest attempt (the stored response is overwritten the
        same way), so its incidentals are that attempt's alone. Appending left
        every attempt's captures under the parent, and an ``incidental=``
        ``Singular`` then found one match per attempt. An empty batch clears
        the parent. Storage rows the deleted captures shared are left: other
        captures may point at them.

        One lock acquisition / session / commit for the whole batch instead
        of one per subresource — a page with dozens of captured sub-requests
        was paying dozens of commits per navigation.

        Each record's payload is deduplicated on the compressed content's
        MD5 and nothing else — ``incidental_request_storage`` is a
        content-addressed store, so the lookup is a single-row probe of a
        unique index. Dedup holds within the batch too: the probe sees
        earlier records' flushed rows.

        Everything that distinguishes one *fetch* from another —
        resource_type, method, url, status_code, failure_reason, the request
        ``body`` that disambiguates several requests to the same URL — is a
        column of ``incidental_requests`` and is written per occurrence, so
        two captures sharing a payload still report their own metadata
        rather than the first writer's. That is what lets the storage key be
        the hash alone.

        It has to be the hash alone. When the key also covered the fetch
        metadata, a site that stamps its asset URLs with the render time
        (``app.js?v=20260918082331``) defeated it outright: every page load
        presented a new URL, so the probe walked every row sharing the
        content hash, missed, and appended one more. The walk grew by a row
        per page load and the run went quadratic — a real corpus reached
        369k rows for a single unchanging jQuery bundle and minutes of
        random I/O per navigation.

        A capture with no body gets no storage row at all; its
        ``storage_id`` stays NULL.

        ``response_headers_json`` is the one field kept on the shared row,
        and is deliberately *not* part of the key: volatile headers
        (``Date``, ``ETag``) differ on nearly every response, so keying on
        them would defeat the dedup as thoroughly as the URL did. A
        deduplicated capture therefore reads back the first capture's
        headers — treat stored response headers as representative, not exact
        (a promotion into a pre-resolved request replays them).

        Returns:
            The incidental_requests row IDs, in record order.
        """
        async with self._write_session() as session:
            await session.execute(
                delete(IncidentalRequest).where(
                    IncidentalRequest.parent_request_id == parent_request_id
                )
            )
            ids = [
                await self._insert_incidental_in_session(
                    session, parent_request_id, record
                )
                for record in records
            ]
            await session.commit()
            return ids

    async def _insert_incidental_in_session(
        self,
        session: AsyncSession,
        parent_request_id: int,
        record: IncidentalCapture,
    ) -> int:
        """Dedup-and-insert one incidental inside an existing session (no commit)."""
        storage_id: int | None = None

        # Content addressing: the digest is the storage row's whole identity,
        # so this is a probe of a unique index, not a scan of everything
        # sharing a hash. No body, no storage row.
        if record.content_compressed is not None:
            content_md5 = record.content_md5
            if content_md5 is None:
                raise ValueError(
                    f"incidental capture of {record.url} has a body but no "
                    "content_md5; record it with IncidentalCapture.set_body"
                )
            result = await session.execute(
                select(IncidentalRequestStorage.id).where(
                    IncidentalRequestStorage.content_md5 == content_md5
                )
            )
            storage_id = result.scalar_one_or_none()

            if storage_id is None:
                storage = IncidentalRequestStorage(
                    **record.model_dump(include=_STORAGE_FIELDS)
                )
                session.add(storage)
                await session.flush()
                storage_id = storage.id

        ir = IncidentalRequest(
            **record.model_dump(include=_REQUEST_FIELDS),
            parent_request_id=parent_request_id,
            storage_id=storage_id,
        )
        session.add(ir)
        await session.flush()
        return ir.id

    async def get_incidental_requests(
        self, parent_request_id: int
    ) -> list[IncidentalRequestRecord]:
        """Get all incidental requests for a parent request.

        Joins with storage table to include content/response fields.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                incidental_record_select()
                .where(
                    IncidentalRequest.parent_request_id == parent_request_id
                )
                .order_by(IncidentalRequest.started_at_ns.asc())
            )
            return [
                IncidentalRequestRecord.model_validate(row)
                for row in result.all()
            ]

    async def get_incidental_request_by_id(
        self, incidental_id: int
    ) -> IncidentalRequestRecord | None:
        """Get a single incidental request by ID with storage data."""
        async with self.session_factory() as session:
            result = await session.execute(
                incidental_record_select().where(
                    IncidentalRequest.id == incidental_id
                )
            )
            row = result.first()
            return (
                IncidentalRequestRecord.model_validate(row)
                if row is not None
                else None
            )

    async def get_incidental_request_body(
        self, incidental_id: int
    ) -> bytes | None:
        """The request body the browser sent on one capture.

        Its own accessor rather than a field of
        :class:`IncidentalRequestRecord`: the body is bytes, the record's
        ``model_dump(mode="json")`` is a display shape, and only
        ``incidental=``'s ``body_contains`` matching ever wants it.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(IncidentalRequest.body).where(
                    IncidentalRequest.id == incidental_id
                )
            )
            return result.scalar_one_or_none()

    async def get_incidental_request_storage(
        self, storage_id: int
    ) -> IncidentalRequestStorage | None:
        """Get the raw storage row (compressed body, headers, dict id)."""
        async with self.session_factory() as session:
            return await session.get(IncidentalRequestStorage, storage_id)
