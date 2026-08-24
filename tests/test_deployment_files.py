import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "docker" / "Dockerfile.cuda"
COMPOSE = ROOT / "docker" / "docker-compose.gpu.yml"


def test_dockerfile_copies_only_paths_that_exist():
    for line in DOCKERFILE.read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        parts = line.split()[1:]
        for source in parts[:-1]:
            if source.startswith("--"):
                continue
            assert (ROOT / source).exists(), source


def test_dockerfile_does_not_reinstall_torch():
    """torch comes from the CUDA base image; reinstalling risks a CPU-only wheel."""
    text = DOCKERFILE.read_text()
    assert "pip install" in text
    assert not re.search(r"pip install[^\n]*\btorch\b", text)


def test_gpu_requirements_exclude_torch_and_include_kokoro():
    text = (ROOT / "requirements-gpu.txt").read_text()
    assert "kokoro==0.9.4" in text
    assert not re.search(r"^torch[=<>]", text, re.MULTILINE)


def test_image_binds_all_interfaces():
    """A 127.0.0.1 bind inside the container is unreachable from the host."""
    assert "KOKORO_HOST=0.0.0.0" in DOCKERFILE.read_text()
    assert "KOKORO_HOST: 0.0.0.0" in COMPOSE.read_text()


def test_compose_reserves_the_gpu():
    text = COMPOSE.read_text()
    assert "driver: nvidia" in text
    assert "capabilities: [gpu]" in text
    assert "KOKORO_DEVICE: cuda" in text


def test_compose_is_valid_yaml_with_one_service():
    try:
        import yaml
    except ImportError:
        import pytest

        pytest.skip("pyyaml not installed")
    config = yaml.safe_load(COMPOSE.read_text())
    assert list(config["services"]) == ["kokoro"]
    assert config["services"]["kokoro"]["ports"] == ["8080:8080"]


def test_dockerignore_excludes_the_local_venv():
    text = (ROOT / ".dockerignore").read_text()
    assert ".venv" in text


def test_the_image_ships_the_vendored_docs_assets():
    """/docs and /redoc are blank in the container without these.

    They are the one place a few MB of committed JS is the right call: the
    container has no network at runtime.
    """
    assert "COPY web ./web" in DOCKERFILE.read_text()
    patterns = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for pattern in patterns:
        assert "vendor" not in pattern, pattern
        assert not pattern.startswith("web"), pattern
    for filename in ("swagger-ui-bundle.js", "swagger-ui.css", "redoc.standalone.js"):
        assert (ROOT / "web" / "vendor" / filename).is_file(), filename


def test_readme_documents_both_deployments():
    text = (ROOT / "README.md").read_text()
    assert "setup_mac.sh" in text
    assert "docker-compose.gpu.yml" in text
    assert "/v1/audio/speech" in text
