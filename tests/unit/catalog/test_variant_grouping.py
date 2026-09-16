from pathlib import Path

from playlist_sorter.acquisition import acquisition_id
from playlist_sorter.catalog import CatalogSong, group_musical_variants, map_catalog_song


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


def test_catalog_mapper_applies_acquisition_metadata_only_for_exact_id_and_hash():
    source_hash = "a" * 64
    acquisition_key = acquisition_id("https://www.youtube.com/watch?v=BaW_jenozKc")
    song = CatalogSong(
        source_hash,
        f"C:/library/{acquisition_key}.flac",
        source_hash,
        123,
        180.0,
        "flac",
        {},
    )
    receipt = {
        "acquisition_id": acquisition_key,
        "url": "https://www.youtube.com/watch?v=BaW_jenozKc",
        "sha256": source_hash,
        "title": "Authorized title",
        "artist": "Authorized artist",
        "source": "youtube",
    }

    mapped = map_catalog_song(song, {acquisition_key: receipt})
    mismatched = map_catalog_song(song, {acquisition_key: {**receipt, "sha256": "b" * 64}})

    assert mapped.integrity_fingerprint == source_hash
    assert mapped.source_path == song.source_path
    assert mapped.file_size == 123
    assert mapped.title == "Authorized title"
    assert mapped.artist == "Authorized artist"
    assert mapped.acquisition is not None
    assert mismatched.title == ""
    assert mismatched.artist == ""
    assert mismatched.acquisition is None


def test_catalog_mapper_rejects_metadata_when_the_receipt_key_or_id_is_wrong():
    song = CatalogSong(
        "a" * 64,
        "C:/library/youtube-0123456789abcdef.flac",
        "a" * 64,
        1,
        None,
        "flac",
        {},
    )
    receipt = {
        "acquisition_id": "youtube-fedcba9876543210",
        "url": "https://www.youtube.com/watch?v=BaW_jenozKc",
        "sha256": "a" * 64,
        "title": "Must not leak",
        "artist": "Must not leak",
        "source": "youtube",
    }

    mapped = map_catalog_song(song, {Path(song.source_path).stem: receipt})

    assert mapped.title == ""
    assert mapped.artist == ""
    assert mapped.acquisition is None


def test_catalog_mapper_rejects_matching_filename_when_url_identity_disagrees():
    acquisition_id = "youtube-0123456789abcdef"
    song = CatalogSong(
        "a" * 64,
        f"C:/library/{acquisition_id}.flac",
        "a" * 64,
        1,
        None,
        "flac",
        {},
    )
    receipt = {
        "acquisition_id": acquisition_id,
        "url": "https://www.youtube.com/watch?v=BaW_jenozKc",
        "sha256": "a" * 64,
        "title": "Must not leak",
        "artist": "Must not leak",
        "source": "youtube",
    }

    mapped = map_catalog_song(song, {acquisition_id: receipt})

    assert mapped.title == ""
    assert mapped.artist == ""
    assert mapped.acquisition is None
