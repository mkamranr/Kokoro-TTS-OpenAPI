import pytest

from app.voices import (
    MAX_BLEND_COMPONENTS,
    OPENAI_ALIASES,
    VOICES,
    VOICES_BY_ID,
    canonical_spec,
    catalog,
    lang_for,
    parse_voice_spec,
    resolve_alias,
)


def test_catalog_has_all_28_english_voices():
    assert len(VOICES) == 28
    assert len({v.id for v in VOICES}) == 28
    assert sum(1 for v in VOICES if v.lang == "a") == 20
    assert sum(1 for v in VOICES if v.lang == "b") == 8


def test_af_heart_is_the_single_default():
    defaults = [v.id for v in VOICES if v.default]
    assert defaults == ["af_heart"]


def test_every_id_matches_its_lang_and_gender_prefix():
    for v in VOICES:
        assert v.id[0] == v.lang
        assert v.id[1] == ("f" if v.gender == "female" else "m")
        assert v.accent == ("American" if v.lang == "a" else "British")


def test_catalog_dicts_are_json_ready():
    rows = catalog()
    assert len(rows) == 28
    assert set(rows[0]) == {
        "id", "name", "gender", "accent", "lang", "grade", "default",
    }


def test_openai_aliases_point_at_real_voices():
    assert set(OPENAI_ALIASES) == {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    }
    for target in OPENAI_ALIASES.values():
        assert target in VOICES_BY_ID
    assert resolve_alias("shimmer") == "af_sky"
    assert resolve_alias("af_bella") == "af_bella"


def test_single_voice_spec():
    comps = parse_voice_spec("af_bella")
    assert len(comps) == 1
    assert comps[0].voice_id == "af_bella"
    assert comps[0].weight == pytest.approx(1.0)


def test_unweighted_blend_averages_equally():
    comps = parse_voice_spec("af_bella,af_sky")
    assert [c.voice_id for c in comps] == ["af_bella", "af_sky"]
    assert [c.weight for c in comps] == pytest.approx([0.5, 0.5])


def test_weighted_blend_normalizes_to_one():
    comps = parse_voice_spec("af_bella:3,af_sky:1")
    assert [c.weight for c in comps] == pytest.approx([0.75, 0.25])


def test_weights_that_already_sum_to_one_are_preserved():
    comps = parse_voice_spec("af_bella:0.6,af_sky:0.4")
    assert [c.weight for c in comps] == pytest.approx([0.6, 0.4])


def test_alias_inside_a_blend_is_resolved():
    assert [c.voice_id for c in parse_voice_spec("nova,onyx")] == [
        "af_nova", "am_onyx",
    ]


def test_canonical_spec_is_stable_for_caching():
    assert canonical_spec(parse_voice_spec("af_bella:0.6,af_sky:0.4")) == (
        "af_bella:0.6000,af_sky:0.4000"
    )


def test_lang_comes_from_the_first_component():
    assert lang_for(parse_voice_spec("af_bella")) == "a"
    assert lang_for(parse_voice_spec("bm_george,af_bella")) == "b"


@pytest.mark.parametrize(
    "spec,fragment",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("nope", "Unknown voice"),
        ("af_bella,nope", "Unknown voice"),
        ("af_bella:0.5,af_sky", "either all"),
        ("af_bella:0", "greater than 0"),
        ("af_bella:-1", "greater than 0"),
        ("af_bella:abc", "not a number"),
        ("af_bella,af_sky,af_nova,am_puck,am_echo", "at most 4"),
        ("zf_xiaobei", "Unknown voice"),
    ],
)
def test_invalid_specs_raise_valueerror(spec, fragment):
    with pytest.raises(ValueError) as excinfo:
        parse_voice_spec(spec)
    assert fragment in str(excinfo.value)


def test_blend_component_cap_constant():
    assert MAX_BLEND_COMPONENTS == 4
