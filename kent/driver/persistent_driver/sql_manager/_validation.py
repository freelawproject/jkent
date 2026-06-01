"""Response validation operations for SQLManager."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import select

from kent.driver.persistent_driver.models import Request

if TYPE_CHECKING:
    import asyncio

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class ValidationMixin:
    """JSON and XML response validation operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

    async def _validate_responses_with(
        self,
        continuation: str,
        validator: Callable[[bytes], bool | None],
    ) -> list[int]:
        """Decompress each response for ``continuation`` and run ``validator(content)``.

        Returns the request_ids where ``validator`` raised or returned ``False``.
        Responses without compressed content are skipped.
        """
        from kent.driver.persistent_driver.compression import (
            decompress_response,
        )

        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Request.id,
                    Request.content_compressed,
                    Request.compression_dict_id,
                ).where(
                    Request.continuation == continuation,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            rows = result.all()

        invalid_request_ids: list[int] = []
        for row in rows:
            request_id, compressed_content, dict_id = row
            if compressed_content is None:
                continue
            try:
                content = await decompress_response(
                    self._session_factory,
                    compressed_content,
                    dict_id,
                )
                if validator(content) is False:
                    invalid_request_ids.append(request_id)
            except Exception:
                invalid_request_ids.append(request_id)

        return invalid_request_ids

    # --- JSON Response Validation ---

    async def validate_json_responses(
        self,
        continuation: str,
        model: type[BaseModel],
    ) -> list[int]:
        """Validate stored JSON responses against a Pydantic model.

        Args:
            continuation: The continuation method name to filter responses.
            model: Pydantic BaseModel class to validate against.

        Returns:
            List of request_id values for responses that failed validation.
        """

        def validate(content: bytes) -> None:
            model.model_validate(json.loads(content.decode("utf-8")))

        return await self._validate_responses_with(continuation, validate)

    # --- XML/XSD Response Validation ---

    async def validate_xml_responses(
        self,
        continuation: str,
        xsd_path: str,
    ) -> list[int]:
        """Validate stored HTML responses against an XSD schema.

        Args:
            continuation: The continuation method name to filter responses.
            xsd_path: Absolute path to the XSD schema file.

        Returns:
            List of request_id values for responses that failed validation.
        """
        from lxml import etree
        from lxml import html as lxml_html

        schema = etree.XMLSchema(etree.parse(xsd_path))  # noqa: S320

        def validate(content: bytes) -> bool:
            return bool(schema.validate(lxml_html.fromstring(content)))

        return await self._validate_responses_with(continuation, validate)
