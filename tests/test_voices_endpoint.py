def test_voices_endpoint_lists_all_28(client):
    body = client.get("/voices").json()
    assert body["count"] == 28
    assert len(body["voices"]) == 28
    assert body["default"] == "af_heart"


def test_voice_rows_carry_display_metadata(client):
    row = next(v for v in client.get("/voices").json()["voices"] if v["id"] == "bm_george")
    assert row["accent"] == "British"
    assert row["gender"] == "male"
    assert row["lang"] == "b"
    assert row["grade"] == "C"
