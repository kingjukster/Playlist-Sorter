from playlist_sorter.catalog import CatalogSong, group_musical_variants


def _song(song_id: str, fingerprint: str | None, duration: float) -> CatalogSong:
    return CatalogSong(song_id, song_id, song_id, 1, duration, "flac", {}, fingerprint, "available")


def test_distinct_encodes_group_only_with_matching_fingerprint_and_duration():
    grouped = group_musical_variants(
        [
            _song("a", "same", 180.0),
            _song("b", "same", 181.0),
            _song("c", "same", 240.0),
            _song("d", None, 180.0),
        ]
    )

    assert grouped[0].song_id != grouped[1].song_id
    assert grouped[0].variant_group_id == grouped[1].variant_group_id
    assert grouped[2].variant_group_id != grouped[0].variant_group_id
    assert grouped[3].variant_group_id == "exact-d"
