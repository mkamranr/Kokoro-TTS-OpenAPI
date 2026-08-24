import re
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def test_index_is_served_at_the_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Kokoro" in resp.text


def test_assets_are_served(client):
    css = client.get("/styles.css")
    js = client.get("/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "/tts/stream" in js.text


def test_every_referenced_asset_exists_on_disk():
    html = (WEB_DIR / "index.html").read_text()
    referenced = re.findall(r'(?:src|href)="(?!https?:|//)([^"]+)"', html)
    assert referenced, "expected local asset references"
    for asset in referenced:
        assert (WEB_DIR / asset.lstrip("/")).is_file(), asset


def test_the_ui_pulls_in_no_remote_resources():
    """The container has no network; a CDN reference would break it."""
    html = (WEB_DIR / "index.html").read_text()
    assert "http://" not in html
    assert "https://" not in html


def test_mounting_the_ui_does_not_shadow_the_api(client):
    assert client.get("/voices").status_code == 200
    assert client.post("/tts", json={"text": "hi"}).status_code == 200
    assert client.get("/health").status_code == 200
