# AGENTS.md — sdypy-view contributor & agent guide

Guidance for anyone (human or AI) developing **sdypy-view**. Claude Code reads
it via a one-line `CLAUDE.md` (`@AGENTS.md`). The org-wide rules (development
workflow, SEP governance, naming, canonical specs) live in the sdypy hub and are
linked at the end, never restated here.

## What this package is

3D visualisation: the `view` portion of the `sdypy` namespace (PEP 420,
`from sdypy import view`). It wraps PyVista and pyvistaqt (`Plotter3D`) with
helpers for finite element meshes, animated displacements and fields, and
copying the image to the clipboard. The public API is the curated `__all__` in
`sdypy/view/__init__.py`.

## Development environment

Use `uv`. Supported Python versions: `requires-python` in `pyproject.toml`.

```console
uv venv && uv pip install -e ".[dev,qt]"   # dev install; `qt` adds the Qt binding
QT_QPA_PLATFORM=offscreen pytest           # headless run: no window opens
python -m build                            # sdist + wheel
```

`import sdypy.view` needs a Qt binding (known issue): without the `qt` extra
the import fails, so the tests cannot be collected. With PyQt6 installed, one
test is skipped (the absent-PyQt6 guard).

## Org-wide rules (sdypy hub)

Copied verbatim from the hub's `AGENTS.md`; edit it there. The hub's template
checker compares this copy.

<!-- >>> sdypy hub links -->
- [AGENTS.md](https://github.com/sdypy/sdypy/blob/main/AGENTS.md) — workflow and definition of done
- [AGENTS.md § Releasing](https://github.com/sdypy/sdypy/blob/main/AGENTS.md#releasing) — the only way to release
- [docs/seps/](https://github.com/sdypy/sdypy/tree/main/docs/seps/) — SEP governance docs
- [docs/source/dev/nomenclature.rst](https://github.com/sdypy/sdypy/blob/main/docs/source/dev/nomenclature.rst) — naming a new public term (SEP 2)
- [openspec/specs/](https://github.com/sdypy/sdypy/tree/main/openspec/specs/) — canonical capability specs
- [REQUIREMENTS.md](https://github.com/sdypy/sdypy/blob/main/REQUIREMENTS.md) — requirements roster
<!-- <<< sdypy hub links -->
