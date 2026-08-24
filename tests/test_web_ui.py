import re
from pathlib import Path

from starlette.routing import Mount

from app.main import create_app

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


def test_the_static_mount_is_registered_last():
    """The invariant that keeps the "/" mount from swallowing the whole API.

    A mount at "/" matches every path beneath it, so any route registered
    after it is unreachable. Asserted structurally, not behaviourally: a route
    added below the mount in create_app() fails HERE, with an explanation,
    instead of silently 404ing in production.
    """
    routes = create_app().routes
    mounts = [route for route in routes if isinstance(route, Mount)]
    assert len(mounts) == 1, f"expected exactly one mount, got {mounts}"
    assert mounts[0].path == ""  # Starlette stores a "/" mount as ""
    after = routes[routes.index(mounts[0]) + 1 :]
    assert not after, (
        "the StaticFiles mount must be the LAST entry in app.routes, but these "
        f"are registered after it and can never match: {after}"
    )
