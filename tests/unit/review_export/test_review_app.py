"""The Streamlit file is an import-safe, localhost-only presentation layer."""


def test_review_app_import_does_not_start_a_server_and_exposes_localhost_contract():
    from playlist_sorter import review_app

    assert review_app.LOCALHOST_ADDRESS == "127.0.0.1"
    assert review_app.review_server_args() == ("--server.address", "127.0.0.1")
    assert "--server.address" in review_app.review_server_command("C:/run")
    assert "127.0.0.1" in review_app.review_server_command("C:/run")
