"""/docs and /redoc must work with no network at all.

FastAPI's stock docs pages load Swagger UI and ReDoc from cdn.jsdelivr.net
(and ReDoc pulls Google Fonts too), which renders both pages blank inside the
offline GPU container. app/main.py points them at assets vendored under
web/vendor by scripts/fetch_docs_assets.py. These tests are the regression
guard: a single reintroduced remote URL fails them.
"""
from pathlib import Path

import pytest

VENDOR_DIR = Path(__file__).resolve().parent.parent / "web" / "vendor"

# Every host the FastAPI defaults would have reached for.
FORBIDDEN_HOSTS = ("cdn.jsdelivr", "fonts.googleapis", "fastapi.tiangolo.com")

VENDORED_FILES = (
    "swagger-ui-bundle.js",
    "swagger-ui.css",
    "redoc.standalone.js",
)


@pytest.fixture(params=["/docs", "/redoc"])
def docs_page(request, client):
    resp = client.get(request.param)
    assert resp.status_code == 200, request.param
    return resp


def test_docs_pages_render(docs_page):
    assert docs_page.headers["content-type"].startswith("text/html")
    assert "<!DOCTYPE html>" in docs_page.text


def test_docs_pages_reference_no_remote_host(docs_page):
    for host in FORBIDDEN_HOSTS:
        assert host not in docs_page.text, host


def test_docs_pages_reference_no_remote_url_at_all(docs_page):
    """Stricter than the host list: no absolute http(s) URL of any kind."""
    assert "http://" not in docs_page.text
    assert "https://" not in docs_page.text


def test_swagger_ui_uses_the_vendored_bundle(client):
    html = client.get("/docs").text
    assert "/vendor/swagger-ui-bundle.js" in html
    assert "/vendor/swagger-ui.css" in html


def test_redoc_uses_the_vendored_bundle(client):
    html = client.get("/redoc").text
    assert "/vendor/redoc.standalone.js" in html


def test_docs_favicon_is_an_inline_data_uri(docs_page):
    assert "data:image/svg+xml;base64," in docs_page.text


def test_openapi_schema_is_served(client):
    """Both pages are useless without it."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/tts" in resp.json()["paths"]


@pytest.mark.parametrize("filename", VENDORED_FILES)
def test_vendored_assets_are_committed_and_plausible(filename):
    path = VENDOR_DIR / filename
    assert path.is_file(), f"{filename} missing; run scripts/fetch_docs_assets.py"
    assert path.stat().st_size > 50 * 1024, filename


def test_vendored_swagger_bundle_is_served_over_http(client):
    resp = client.get("/vendor/swagger-ui-bundle.js")
    assert resp.status_code == 200
    assert len(resp.content) > 100 * 1024, len(resp.content)


@pytest.mark.parametrize("filename", VENDORED_FILES)
def test_every_vendored_asset_is_reachable(client, filename):
    resp = client.get(f"/vendor/{filename}")
    assert resp.status_code == 200, filename
    assert len(resp.content) > 50 * 1024, filename


def test_vendored_css_embeds_its_own_images_and_fonts():
    """A url(https://...) in the CSS would be a silent remote fetch."""
    css = (VENDOR_DIR / "swagger-ui.css").read_text(encoding="utf-8")
    assert "https://" not in css
    assert "fonts.gstatic" not in css


def test_docs_are_hidden_from_the_openapi_schema(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/docs" not in paths
    assert "/redoc" not in paths


def test_fetch_script_pins_exact_versions():
    """A floating version would make the vendored bytes unreproducible."""
    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "fetch_docs_assets.py"
    ).read_text(encoding="utf-8")
    assert 'SWAGGER_UI_VERSION = "5.32.14"' in source
    assert 'REDOC_VERSION = "2.5.3"' in source
