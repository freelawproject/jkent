"""Request-prep provider ABCs.

Operators implement these to provide captcha solvers and similar preprocessors
to the driver via ``Driver.open(request_preps=[...])``. Scraper code yields
``JSRequestPrep`` / ``HTTPRequestPrep`` wrappers with ``prep_method="provided.X"``
to dispatch to a registered provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from jkent.data_types import BaseScraper, DriverRequirement

if TYPE_CHECKING:
    from playwright.async_api import Page

    from jkent.data_types import Request, Response


class RequestPrepProvider(ABC):
    """Marker base for anything that can register as a driver-provided prep.

    Concrete subclasses (one per "kind" of prep — see ``HCaptchaSolver``,
    ``ImageCaptchaSolver``) declare a stable ``provider_name`` and a
    ``requires_live_page`` flag. Operators subclass the concrete kind to
    provide an implementation; the namespace stays stable across operator
    swaps because ``provider_name`` is on the jkent-shipped ABC.
    """

    provider_name: ClassVar[str]
    # When True, only drivers with a live Playwright Page can use this provider
    # (httpx driver rejects them at open() time).
    requires_live_page: ClassVar[bool] = False

    # Concrete subclasses (HCaptchaSolver / ImageCaptchaSolver) declare the
    # prep signature shape; an attribute reference here is enough for static
    # callers like ``build_provided_preps``.
    prep: Callable[..., Any]


class HCaptchaHandler(RequestPrepProvider, ABC):
    """JS-driven hCaptcha token producer.

    The prep runs against the live parent page and is expected to call
    ``page.evaluate(...)`` to obtain a token, then bake it into the
    request's headers (or query string, as the operator chooses).
    """

    provider_name: ClassVar[str] = "hcaptcha_handler"
    requires_live_page: ClassVar[bool] = True

    @abstractmethod
    async def prep(
        self,
        response: Response,
        request: Request,
        page: Page,
    ) -> Request: ...


class ImageCaptchaSolver(RequestPrepProvider, ABC):
    """HTTP-driven image-captcha solver.

    The prep fetches the captcha image, hands it to a paid solver service,
    and bakes the answer into the form data of the inner Request.
    """

    provider_name: ClassVar[str] = "image_captcha_handler"
    requires_live_page: ClassVar[bool] = False

    @abstractmethod
    async def prep(
        self,
        response: Response,
        request: Request,
        **kwargs: Any,
    ) -> Request: ...


# Map DriverRequirement enum members → provider_name. Used to validate at
# Driver.open() time that any prep-related requirements declared by the
# scraper are satisfied by something in ``request_preps=[]``.
REQUIREMENT_TO_PROVIDED_PREP: dict[DriverRequirement, str] = {
    DriverRequirement.HCAP_HANDLER: HCaptchaHandler.provider_name,
    DriverRequirement.IMAGE_CAPTCHA_HANDLER: ImageCaptchaSolver.provider_name,
}


class WordImageCaptcha(ImageCaptchaSolver):
    """ImageCaptchaSolver that posts the image to an external OCR service.

    The service is expected to accept an HTTP POST to ``server_url`` with
    the image bytes attached as multipart form-data under field name
    ``image``, and to respond with the recognized text as the plain-text
    response body. ``thebes/resolve.py`` (a SmolVLM-Instruct wrapper) is
    a working reference implementation; any equivalent service works.

    The prep accepts two kwargs at the yield site:
    - ``image_url``: URL of the captcha image to fetch.
    - ``result_field``: form-data key to populate with the answer.

    Example yield site::

        yield HTTPRequestPrep(
            Request(
                request=HTTPRequestParams(
                    method=HttpMethod.POST,
                    url="https://example.com/login",
                    data={"mode": "edit", "embedded": token, "task": "DOCKET"},
                ),
                continuation=self.parse_search_page,
            ),
            prep_method="provided.image_captcha_handler",
            image_url="https://example.com/captcha.png?session=abc",
            result_field="captchaEntry",
        )
    """

    def __init__(self, server_url: str) -> None:
        """
        Args:
            server_url: Resolver endpoint that accepts a POST with the
                image as multipart form-data (field name ``image``) and
                returns the recognized text as the plain-text body.
        """
        self.server_url = server_url

    async def prep(
        self,
        response: Response,
        request: Request,
        **kwargs: Any,
    ) -> Request:
        try:
            image_url: str = kwargs["image_url"]
            result_field: str = kwargs["result_field"]
        except KeyError as e:
            raise TypeError(
                "WordImageCaptcha.prep requires kwargs "
                "'image_url' and 'result_field'"
            ) from e

        async with httpx.AsyncClient() as client:
            img_resp = await client.get(image_url)
            img_resp.raise_for_status()
            image_bytes = img_resp.content

            files = {"image": ("captcha.png", image_bytes, "image/png")}
            solver_resp = await client.post(self.server_url, files=files)
            solver_resp.raise_for_status()
            answer = solver_resp.text.strip()

        existing = request.request.data
        existing_dict: dict[str, Any] = (
            dict(existing) if isinstance(existing, dict) else {}
        )
        new_data = {**existing_dict, result_field: answer}
        new_http = replace(request.request, data=new_data)
        return replace(request, request=new_http)


def build_provided_preps(
    scraper: BaseScraper,
    request_preps: list[RequestPrepProvider] | None,
    *,
    allow_live_page_providers: bool,
) -> dict[str, Callable[..., Any]]:
    """Build the ``provider_name → prep`` dispatch table for a driver.

    Validates that:
    - No two instances claim the same ``provider_name``
    - When ``allow_live_page_providers`` is False, no provider has
      ``requires_live_page=True`` (rejects JS-flavored providers under
      the httpx driver)
    - Every prep-related ``DriverRequirement`` declared by the scraper
      is satisfied by a provider in the list

    Raises:
        ValueError: For any of the above violations.
    """
    provided: dict[str, Callable[..., Any]] = {}
    for instance in request_preps or []:
        name = instance.provider_name
        if not allow_live_page_providers and instance.requires_live_page:
            raise ValueError(
                f"request_prep {type(instance).__name__!r} "
                f"(provider_name={name!r}) requires a live Playwright Page "
                f"and cannot be used with this driver"
            )
        if name in provided:
            raise ValueError(
                f"duplicate provider_name {name!r} among request_preps "
                f"(at least two instances claim it)"
            )
        provided[name] = instance.prep

    for req in scraper.driver_requirements:
        key = REQUIREMENT_TO_PROVIDED_PREP.get(req)
        if key and key not in provided:
            raise ValueError(
                f"scraper requires {req.name} but no request_prep with "
                f"provider_name={key!r} was passed to open()"
            )

    return provided
