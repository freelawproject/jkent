# `@entry` and `@step`

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/common/decorators.py, jkent/common/decorator_metadata.py
     Delete this comment when the page has real content. -->

The two decorators that define a scraper's shape, and the parameters a step may take.

## Planned contents

- `@entry(T)` signature and return type
- `@step` signature, overloads, priority
- The full injection table: response, request, previous_request, accumulated_data, json_content, lxml_tree, page, text, local_filepath
- `register_injector`, `InjectionContext`, `Injector`
- `DEFAULT_PRIORITY`
