"""Pre-download the model, config, and all 28 English voice packs.

Used by the Mac setup script and by the Docker build, so a container can start
with no network access.
"""
import os
import sys

sys.path.insert(0, ".")
import app  # noqa: E402  (must import before huggingface_hub: sets HF_HOME default)
from app.voices import VOICES  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "hexgrad/Kokoro-82M"


def main() -> None:
    print(f"HF_HOME={os.environ['HF_HOME']}")
    for filename in ("config.json", "kokoro-v1_0.pth"):
        print(f"fetching {filename}")
        hf_hub_download(REPO_ID, filename)
    for voice in VOICES:
        print(f"fetching voices/{voice.id}.pt")
        hf_hub_download(REPO_ID, f"voices/{voice.id}.pt")
    print(f"done: model + {len(VOICES)} English voices cached")


if __name__ == "__main__":
    main()
