# Releasing pyBer

Releases are built and published by GitHub Actions
(`.github/workflows/build-and-publish.yml`). Nobody needs to build or upload
executables by hand.

## To publish a new version

1. Set the new version in `pyBer/version.py`, e.g. `__version__ = "0.58.0"`.
2. Write the release notes in `docs/releases/<tag>.md`. The tag drops a `.0`
   patch: 0.58.0 → `v0.58.md`, 0.58.1 → `v0.58.1.md`.
3. Update the README download links and the "What changed" section.
4. Commit and push to `main`.

That push runs the full pipeline (about 16 minutes). If every step passes and
the version has no release yet, the release is published automatically with
`pyBer-windows.exe`, `pyBer-cli-windows.exe` and `SHA256SUMS.txt`. Pushing a
tag by hand still works, but it must match `version.py`.

## What every push to `main` checks

| Step | Time | Fails when |
|---|---|---|
| Conda environment (pinned, cached) | ~3 min | dependencies cannot be resolved |
| Regression tests | ~6 min | any unit/GUI test fails |
| PyInstaller GUI + CLI | ~5 min | packaging fails |
| Smoke test | <1 min | the executables do not start |
| **Packaged UI audit** | ~1 min | a numeric value is hidden or clipped in any drawer of the *packaged* app |
| Publish (new version only) | ~1 min | release notes missing, or uploaded files differ from the tested build |

The UI audit runs inside the frozen executable (`PYBER_UI_AUDIT=<folder>`), so
it tests the Qt version that is actually shipped. It opens every preprocessing
and postprocessing drawer and subsection and checks every visible numeric
field. Its screenshots and `ui_audit.json` are attached to each run as the
`ui-audit` artifact. It exists because v0.57 shipped with Qt 6.11.3, which hid
every parameter value while all unit tests passed on the developer's Qt.

A failed run never publishes. GitHub emails the person who pushed.

## Running the audit locally

```bash
PYBER_UI_AUDIT=_test/ui_audit python pyBer/main.py   # from source
PYBER_UI_AUDIT=ui_audit dist/pyBer.exe                # packaged build
```

The exit code is non-zero when any field fails.
