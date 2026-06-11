"""Tests for the request-prep providers (jkent.preps).

Covers ``build_provided_preps``'s validation branches and drives the
``WordImageCaptcha`` prep against a stub OCR service (a local aiohttp
server standing in for the resolver endpoint).
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp import web

from jkent.data_types import (
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.preps import (
    HCaptchaHandler,
    ImageCaptchaSolver,
    WordImageCaptcha,
    build_provided_preps,
)


class _Scraper:
    """Scraper stand-in: only ``driver_requirements`` is consulted."""

    def __init__(self, requirements: list[DriverRequirement] | None = None):
        self.driver_requirements = requirements or []


class _OcrSolver(ImageCaptchaSolver):
    async def prep(self, response: Any, request: Any, **kwargs: Any) -> Any:
        return request


class _JsSolver(HCaptchaHandler):
    async def prep(self, response: Any, request: Any, page: Any) -> Any:
        return request


class TestBuildProvidedPreps:
    def test_maps_provider_names(self) -> None:
        solver = _OcrSolver()
        provided = build_provided_preps(
            _Scraper(),  # type: ignore[arg-type]
            [solver],
            allow_live_page_providers=False,
        )
        assert provided == {"image_captcha_handler": solver.prep}

    def test_none_is_empty(self) -> None:
        assert (
            build_provided_preps(
                _Scraper(),  # type: ignore[arg-type]
                None,
                allow_live_page_providers=False,
            )
            == {}
        )

    def test_duplicate_provider_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate provider_name"):
            build_provided_preps(
                _Scraper(),  # type: ignore[arg-type]
                [_OcrSolver(), _OcrSolver()],
                allow_live_page_providers=False,
            )

    def test_live_page_provider_rejected_without_page(self) -> None:
        with pytest.raises(ValueError, match="requires a live Playwright"):
            build_provided_preps(
                _Scraper(),  # type: ignore[arg-type]
                [_JsSolver()],
                allow_live_page_providers=False,
            )
        # ...but accepted when the driver can supply one.
        provided = build_provided_preps(
            _Scraper(),  # type: ignore[arg-type]
            [_JsSolver()],
            allow_live_page_providers=True,
        )
        assert "hcaptcha_handler" in provided

    def test_unmet_requirement_rejected(self) -> None:
        scraper = _Scraper([DriverRequirement.IMAGE_CAPTCHA_HANDLER])
        with pytest.raises(ValueError, match="no request_prep"):
            build_provided_preps(
                scraper,  # type: ignore[arg-type]
                [],
                allow_live_page_providers=False,
            )
        # Satisfied when the matching provider is present.
        provided = build_provided_preps(
            scraper,  # type: ignore[arg-type]
            [_OcrSolver()],
            allow_live_page_providers=False,
        )
        assert "image_captcha_handler" in provided


def _dummy_response(request: Request) -> Response:
    return Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url=request.request.url,
        request=request,
    )


def _login_request() -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/login",
            data={"mode": "edit"},
        ),
        continuation="parse",
    )


async def test_word_image_captcha_bakes_answer_into_form() -> None:
    """The prep fetches the image, posts it to the solver, merges the answer."""
    posted: dict[str, bytes] = {}

    async def serve_image(_request: web.Request) -> web.Response:
        return web.Response(body=b"\x89PNG-fake-captcha")

    async def solve(request: web.Request) -> web.Response:
        form = await request.post()
        field = form["image"]
        assert isinstance(field, web.FileField)
        posted["image"] = field.file.read()
        return web.Response(text="  WORDS42\n")

    app = web.Application()
    app.router.add_get("/captcha.png", serve_image)
    app.router.add_post("/solve", solve)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][0], runner.addresses[0][1]
    base = f"http://{host}:{port}"

    try:
        solver = WordImageCaptcha(server_url=f"{base}/solve")
        original = _login_request()
        prepared = await solver.prep(
            _dummy_response(original),  # unused by this prep
            original,
            image_url=f"{base}/captcha.png",
            result_field="captchaEntry",
        )
    finally:
        await runner.cleanup()

    assert posted["image"] == b"\x89PNG-fake-captcha"
    # Answer stripped and merged into a *copy* of the form data.
    assert prepared.request.data == {
        "mode": "edit",
        "captchaEntry": "WORDS42",
    }
    assert original.request.data == {"mode": "edit"}  # original untouched


async def test_word_image_captcha_requires_kwargs() -> None:
    solver = WordImageCaptcha(server_url="http://irrelevant")
    request = _login_request()
    with pytest.raises(TypeError, match="image_url"):
        await solver.prep(_dummy_response(request), request)
