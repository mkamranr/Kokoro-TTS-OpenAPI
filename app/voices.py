"""The 28 English Kokoro voices, OpenAI name aliases, and blend-spec parsing.

A blend spec is either a single voice id ("af_bella"), a comma-separated list
to average equally ("af_bella,af_sky"), or weighted components
("af_bella:0.6,af_sky:0.4"). Weights are normalized to sum to 1.
"""
from dataclasses import dataclass

DEFAULT_VOICE_ID = "af_heart"
MAX_BLEND_COMPONENTS = 4


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    gender: str
    accent: str
    lang: str
    grade: str
    default: bool = False


def _v(vid: str, name: str, grade: str, default: bool = False) -> Voice:
    gender = "female" if vid[1] == "f" else "male"
    lang = vid[0]
    accent = "American" if lang == "a" else "British"
    return Voice(vid, name, gender, accent, lang, grade, default)


VOICES: tuple[Voice, ...] = (
    _v("af_heart", "Heart", "A", default=True),
    _v("af_bella", "Bella", "A-"),
    _v("af_nicole", "Nicole", "B-"),
    _v("af_aoede", "Aoede", "C+"),
    _v("af_kore", "Kore", "C+"),
    _v("af_sarah", "Sarah", "C+"),
    _v("af_alloy", "Alloy", "C"),
    _v("af_nova", "Nova", "C"),
    _v("af_sky", "Sky", "C-"),
    _v("af_jessica", "Jessica", "D"),
    _v("af_river", "River", "D"),
    _v("am_fenrir", "Fenrir", "C+"),
    _v("am_michael", "Michael", "C+"),
    _v("am_puck", "Puck", "C+"),
    _v("am_echo", "Echo", "D"),
    _v("am_eric", "Eric", "D"),
    _v("am_liam", "Liam", "D"),
    _v("am_onyx", "Onyx", "D"),
    _v("am_santa", "Santa", "D-"),
    _v("am_adam", "Adam", "F+"),
    _v("bf_emma", "Emma", "B-"),
    _v("bf_isabella", "Isabella", "C"),
    _v("bf_alice", "Alice", "D"),
    _v("bf_lily", "Lily", "D"),
    _v("bm_fable", "Fable", "C"),
    _v("bm_george", "George", "C"),
    _v("bm_lewis", "Lewis", "D+"),
    _v("bm_daniel", "Daniel", "D"),
)

VOICES_BY_ID: dict[str, Voice] = {v.id: v for v in VOICES}

# OpenAI's six voice names mapped onto the closest Kokoro voice, so existing
# OpenAI TTS clients work against /v1/audio/speech unchanged.
OPENAI_ALIASES: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}


@dataclass(frozen=True)
class BlendComponent:
    voice_id: str
    weight: float


def resolve_alias(name: str) -> str:
    return OPENAI_ALIASES.get(name.strip().lower(), name.strip())


def catalog() -> list[dict]:
    return [
        {
            "id": v.id,
            "name": v.name,
            "gender": v.gender,
            "accent": v.accent,
            "lang": v.lang,
            "grade": v.grade,
            "default": v.default,
        }
        for v in VOICES
    ]


def parse_voice_spec(spec: str) -> list[BlendComponent]:
    if not spec or not spec.strip():
        raise ValueError("Voice spec is empty")

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Voice spec is empty")
    if len(parts) > MAX_BLEND_COMPONENTS:
        raise ValueError(
            f"A blend may combine at most {MAX_BLEND_COMPONENTS} voices, got {len(parts)}"
        )

    ids: list[str] = []
    weights: list[float | None] = []
    for part in parts:
        name, sep, raw_weight = part.partition(":")
        voice_id = resolve_alias(name)
        if voice_id not in VOICES_BY_ID:
            raise ValueError(
                f"Unknown voice '{name.strip()}'. See GET /voices for the 28 supported ids."
            )
        ids.append(voice_id)
        if not sep:
            weights.append(None)
            continue
        try:
            weight = float(raw_weight)
        except ValueError:
            raise ValueError(
                f"Weight for '{voice_id}' is not a number: '{raw_weight.strip()}'"
            ) from None
        if weight <= 0:
            raise ValueError(f"Weight for '{voice_id}' must be greater than 0")
        weights.append(weight)

    explicit = [w for w in weights if w is not None]
    if explicit and len(explicit) != len(weights):
        raise ValueError(
            "Blend weights must be either all present or all absent, e.g. "
            "'af_bella:0.6,af_sky:0.4' or 'af_bella,af_sky'"
        )

    if not explicit:
        share = 1.0 / len(ids)
        return [BlendComponent(v, share) for v in ids]

    total = sum(explicit)
    return [BlendComponent(v, w / total) for v, w in zip(ids, explicit)]


def canonical_spec(components: list[BlendComponent]) -> str:
    """Stable key for the voice-pack cache."""
    return ",".join(f"{c.voice_id}:{c.weight:.4f}" for c in components)


def lang_for(components: list[BlendComponent]) -> str:
    return components[0].voice_id[0]
