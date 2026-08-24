"""Pre-download the model, config, and all 28 English voice packs.

Used by the Mac setup script and by the Docker build, so a container can start
with no network access.
"""
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from app.voices import VOICES  # noqa: E402

REPO_ID = "hexgrad/Kokoro-82M"


def main() -> None:
    for filename in ("config.json", "kokoro-v1_0.pth"):
        print(f"fetching {filename}")
        hf_hub_download(REPO_ID, filename)
    for voice in VOICES:
        print(f"fetching voices/{voice.id}.pt")
        hf_hub_download(REPO_ID, f"voices/{voice.id}.pt")
    print(f"done: model + {len(VOICES)} English voices cached")


if __name__ == "__main__":
    main()
