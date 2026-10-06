# How text is decoded and why

Httpx and Playwright are both used to fetch content, and both of them do decoding slightly differently. On top of that, lxml does its own decoding of documents when we hand it bytes. We add a small set of helper functions and config points for specifying fallback encoding when servers don't specify it, and the sensible default of utf8 fails.

We have a few goals for this system:

1. We're capturing enough information to debug encoding errors. Primarily if we get garbled text showing up, we want to know if it's something the court was handing us, or an issue with our processing.

2. We want to be able to cover cases where courts send us text encoded one way and advertised another way (say `Content-type: utf-8` in a header, but the content is actually cp1252), or not advertised at all.

3. We want to be able to view and replay original content from the DB in a variety of circumstances, without necessarily synthesizing a full response.

## How the transports handle decoding by default

### HTTPX (if we read `response.text`)

```mermaid
flowchart TD
    H1{"response.encoding set<br/>explicitly?"}
    H2{"Content-Type charset?<br/>(email.message parse)"}
    H3{"a codec Python knows?"}
    H4["default_encoding — 'utf-8',<br/>or a callable autodetector"]
    H5["decode(errors='replace')<br/>incremental, never strict"]
    OUT["str"]
    H1 -->|"yes"| H5
    H1 -->|"no"| H2
    H2 -->|"present"| H3
    H2 -->|"absent"| H4
    H3 -->|"yes"| H5
    H3 -->|"no"| H4
    H4 --> H5
    H5 --> OUT
```

Httpx doesn't read from the doc as part of its decoding, so a
`<meta charset>` is invisible to it. Httpx also does a very permissive
`errors="replace"`, so a wrong header yields garbledygook, or
U+FFFD, and no error to let us know something failed. We store the raw content bytes (zsted)
in the requests table with headers.

### Playwright/Camoufox

This is the HTML standard's encoding sniffing algorithm, which Firefox (so
Camoufox) and Chromium both implement. It runs inside the browser, before any
jkent code exists.

```mermaid
flowchart TD
    B1{"BOM?"}
    B2{"Content-Type charset?"}
    B3{"meta charset in the<br/>first 1024 bytes?"}
    B4{"same-origin parent<br/>browsing context?"}
    B5{"previous visit, or<br/>locale autodetection?"}
    B6["locale default<br/>(e.g. windows-1252)"]
    B7["decode with replacement<br/>into the DOM (UTF-16 str)"]
    OUT["str we can get from playwright"]
    B1 -->|"yes: certain"| B7
    B1 -->|"no"| B2
    B2 -->|"present and known"| B7
    B2 -->|"absent"| B3
    B3 -->|"found"| B7
    B3 -->|"none"| B4
    B4 -->|"yes"| B7
    B4 -->|"no"| B5
    B5 -->|"yes"| B7
    B5 -->|"no"| B6 --> B7
    B7 --> OUT
```

Importantly, when we're using playwright, we're mostly doing it in circumstances where javascript is
adding things to the DOM, so we aren't necessarily interested in parsing the first document we get,
but the document that is being rendered after some short period of time. When we ask playwright for
that serialized DOM, we're always getting it handed to us as a string. We need to pick some encoding for it, so we pick utf8. The dom snapshot is stored with a synthesized header indicating the utf8 encoding and the first charset in the head is rewritten to prevent conflicts. Additionally, we collect the documents that are fetched as part of our browsing as incidental_requests and those don't go through the decoding pipeline of the browser, so we can use those if needed for debugging/inspection.

The edge case here would come from a case where the server responds twice with different content-type headers but the same body (md5). We judge this to be a rare enough occurence that we accept the blindspot here.


## How JKent overrides it

```mermaid
flowchart TD
    subgraph fetch["Fetching"]
        HX["httpx transport<br/>raw wire bytes + server headers"]
        PW["Playwright / Camoufox<br/>rendered DOM, serialized to str"]
        U8["utf8_document()<br/>encode UTF-8, rewrite a stale declaration"]
        SYN["synthesized header<br/>content-type: text/html; charset=utf-8"]
        PW --> U8 --> SYN
    end

    RESP["Response(content=bytes, headers=headers)"]
    HX --> RESP
    SYN --> RESP

    DB[("run database<br/>content_compressed (zstd)<br/>response_headers_json")]
    RESP --> DB
    DB -->|"decompress + stored headers"| REPLAY["Response rebuilt<br/>JKentParser.from_response"]

    subgraph decode["decode_text(content, headers, fallback)"]
        direction TB
        D0{"BOM?"}
        D1{"Content-Type charset?"}
        D2{"markup declaration?<br/>xml encoding=, meta charset"}
        D3{"valid UTF-8?"}
        D4["fallback: the @step encoding<br/>errors='replace'"]
        OUT["str"]
        D0 -->|"decodes strictly"| OUT
        D0 -->|"absent or<br/>bytes rejected"| D1
        D1 -->|"decodes strictly"| OUT
        D1 -->|"absent, unknown codec,<br/>or bytes rejected"| D2
        D2 -->|"decodes strictly"| OUT
        D2 -->|"absent, unknown codec,<br/>or bytes rejected"| D3
        D3 -->|"yes"| OUT
        D3 -->|"no"| D4
        D4 --> OUT
    end

    RESP --> D0
    REPLAY --> D0

```


### Why this precedence

The order is:

1. byte-order mark

2. `Content-Type` header

3. markup declaration (`<?xml encoding=...?>`, `<meta charset>`)

4. strict UTF-8

5. `@step` encoding with replacement (default utf8).

Only the last one always succeeds. Steps 1-4 happen with strict decoding so that
we can cascade further if they fail. Steps 1-3 follow the WHATWG order the browser
uses; we had no observed data justifying a different one.

Strict decoding is where we go further than the browser: a header that names a
charset the bytes don't fit (say `charset=utf-8` over cp1252 bytes) is skipped
rather than applied with replacement, so the markup declaration, UTF-8, or the
`@step` encoding gets a chance at it.