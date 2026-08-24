"""Vendor the Swagger UI and ReDoc assets that /docs and /redoc need.

FastAPI's stock /docs and /redoc pull Swagger UI and ReDoc from
cdn.jsdelivr.net (and ReDoc also pulls Google Fonts). The GPU container has no
network at runtime, so those pages render blank there. This script downloads
PINNED copies into web/vendor/, which the existing StaticFiles mount at "/"
serves as /vendor/... -- no extra mount, no remote request.

The downloaded files are COMMITTED on purpose: a few MB in git is the price of
API docs that work offline.

Re-run this to refresh them after bumping the versions below:

    .venv/bin/python scripts/fetch_docs_assets.py

Idempotent: it always writes the pinned bytes and reports whether each file
changed, so running it twice leaves the tree identical.
"""
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Pinned exactly. FastAPI 0.141.1 defaults to the floating "swagger-ui-dist@5"
# and "redoc@2" ranges; these are the versions those ranges resolved to when
# the assets were vendored, so /docs and /redoc render what the CDN was
# serving -- just from disk.
SWAGGER_UI_VERSION = "5.32.14"
REDOC_VERSION = "2.5.3"

CDN = "https://cdn.jsdelivr.net/npm"
VENDOR_DIR = Path(__file__).resolve().parent.parent / "web" / "vendor"

# (url, filename, minimum plausible size in bytes)
ASSETS = (
    (
        f"{CDN}/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui-bundle.js",
        "swagger-ui-bundle.js",
        100 * 1024,
    ),
    (
        f"{CDN}/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui.css",
        "swagger-ui.css",
        50 * 1024,
    ),
    (
        f"{CDN}/redoc@{REDOC_VERSION}/bundles/redoc.standalone.js",
        "redoc.standalone.js",
        100 * 1024,
    ),
    # NOTE: no favicon file. app/main.py passes a data: URI SVG instead, so
    # there is nothing extra to fetch, ship, or 404 on.
)

TIMEOUT_SECONDS = 60


def ssl_context() -> ssl.SSLContext:
    """Verify against certifi's CA bundle when present.

    A framework python3.10 on macOS often has an empty system trust store, so
    the stdlib default fails with CERTIFICATE_VERIFY_FAILED. certifi is already
    installed (huggingface-hub -> requests depends on it). Verification is
    never disabled -- we are downloading code that will be served to browsers.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, minimum: int) -> bytes:
    """Fetch url, failing loudly on a bad status or an implausibly small body."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "kokoro-tts-api/fetch_docs_assets"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=TIMEOUT_SECONDS, context=ssl_context()
        ) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:  # non-2xx
        raise SystemExit(f"FAILED {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"FAILED {url}: {exc.reason}") from exc

    if status != 200:
        raise SystemExit(f"FAILED {url}: HTTP {status}, expected 200")
    if len(body) < minimum:
        raise SystemExit(
            f"FAILED {url}: got {len(body)} bytes, expected at least {minimum} "
            "-- that is an error page, not the asset"
        )
    return body


def main() -> None:
    print(f"swagger-ui-dist@{SWAGGER_UI_VERSION}  redoc@{REDOC_VERSION}")
    print(f"target: {VENDOR_DIR}")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for url, filename, minimum in ASSETS:
        body = download(url, minimum)
        destination = VENDOR_DIR / filename
        previous = destination.read_bytes() if destination.is_file() else None
        state = "unchanged" if previous == body else "written"
        if previous != body:
            destination.write_bytes(body)
        total += len(body)
        print(f"  {filename:<24} {len(body):>9,} bytes  {state}")

    print(f"done: {len(ASSETS)} files, {total:,} bytes total")


if __name__ == "__main__":
    sys.exit(main())
