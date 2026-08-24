"""Environment-driven settings. Every value has a default."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KOKORO_", case_sensitive=False, extra="ignore"
    )

    device: str = "auto"
    default_voice: str = "af_heart"
    max_chars: int = 5000
    # 0 means "decide from the device": 1 on CPU, 2 on CUDA.
    max_concurrency: int = 0
    # 0 means "leave torch's default alone".
    torch_threads: int = 0
    api_key: str = ""
    voice_cache_size: int = 32
    host: str = "127.0.0.1"
    port: int = 8080
    allow_origins: str = ""

    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.allow_origins.split(",") if o.strip()]


def resolve_concurrency(device: str, configured: int) -> int:
    if configured > 0:
        return configured
    return 2 if device == "cuda" else 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
