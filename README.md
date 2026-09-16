# Playlist Sorter: Apple Music profile exporter

This repository includes a small, local-first CLI that turns Apple Music
playlist data into a validated dataset for later taste analysis. It does not
make recommendations, modify playlists, scrape credentials, or upload listening
data. The larger automatic playlist-discovery roadmap remains in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Outputs

One build writes:

```text
music_profile_source.json
music_profile_summary.md
validation_report.json
```

The canonical JSON contains playlists, tracks, albums, artists, manual album
preferences, objective statistics, cross-playlist duplicates, and album
coverage. With catalog enrichment enabled, albums represented by at least three
tracks also receive an official track listing when Apple returns one.

Generated profiles, private captures, tokens, and `.music-profile/` state are
ignored by Git.

## Install

Python 3.12 or newer is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\music-profile.exe --help
```

The runtime uses only the Python standard library.

## Add private Apple Music data

Apple's web player can expose only a subset of a private playlist. A 100-song
web snapshot is therefore not accepted as complete when the page declares a
larger count. Captures record both observed and declared counts, and validation
fails clearly if they differ.

Use one of these methods.

### Free local browser capture (recommended)

This method requires no Apple Developer membership, API key, subscription
beyond your existing Apple Music account, or third-party service. It reads only
the track rows already rendered in the signed-in web player and downloads JSON
locally; it does not read cookies, tokens, browser storage, or network traffic.

For each playlist:

1. Open the playlist in Apple Music on the web and wait for its first tracks.
2. Open the browser developer tools and select **Console**.
3. Paste the contents of `tools/apple_music_capture.js` and press Enter.
4. Let the script scroll through every lazy-loaded batch, then confirm that the
   alert says the capture is complete.
5. Import the downloaded JSON file.

```powershell
.\.venv\Scripts\music-profile.exe add-playlist `
  "https://music.apple.com/us/library/playlist/p.example" `
  --capture "$env:USERPROFILE\Downloads\apple-music-example.json"
```

Repeat for the primary playlist, Replay All Time, and each yearly Replay, then
run `music-profile build --enrich`. The script waits until the row count and
page height stabilize. If loading does not stabilize or any track lacks its
required title or artist, the capture is marked incomplete and validation fails
rather than silently accepting truncated data. Apple sometimes leaves the page
label at `100 Songs` for much larger private playlists, so completeness uses the
fully loaded DOM row count as the stronger observation.

### MusicKit API with full pagination

This optional route requires paid Apple Developer Program access, a developer
token, and a Music User Token for private-library data. Place existing tokens in
the current process environment:

```powershell
$env:APPLE_MUSIC_DEVELOPER_TOKEN = "<developer-token>"
$env:APPLE_MUSIC_USER_TOKEN = "<music-user-token>"

.\.venv\Scripts\music-profile.exe add-playlist `
  "https://music.apple.com/us/library/playlist/p.example"
```

The CLI follows every Apple `next` page. Tokens are never written or echoed.

Replay URLs use catalog playlist IDs beginning with `pl.rp-`. They need the
developer token; when a Music User Token is present the CLI includes it so Apple
can resolve personalized content:

```powershell
.\.venv\Scripts\music-profile.exe add-playlist `
  "https://music.apple.com/us/playlist/replay-all-time/pl.rp-example"
```

`add-playlist` accepts multiple URLs, so a PowerShell array can fetch Muy fuego
and every Replay in one invocation.

For the locally detected source inventory:

```powershell
$playlistUrls = Get-Content .\.music-profile\detected-playlist-urls.txt |
  Where-Object { $_ -and -not $_.StartsWith("#") }
.\.venv\Scripts\music-profile.exe add-playlist $playlistUrls
.\.venv\Scripts\music-profile.exe build --enrich
```

### Exported library or capture file

JSON, CSV, Apple plist, and iTunes XML are supported:

```powershell
.\.venv\Scripts\music-profile.exe import-file "D:\Exports\Library.xml"
.\.venv\Scripts\music-profile.exe add-playlist `
  "https://music.apple.com/us/library/playlist/p.example" `
  --capture ".\data\muy-fuego.json"
```

Browser-capture JSON is intentionally simple:

```json
{
  "playlist": {
    "playlist_id": "p.example",
    "name": "Example",
    "url": "https://music.apple.com/us/library/playlist/p.example",
    "source_track_count": 785,
    "is_complete": true
  },
  "is_complete": true,
  "source_track_count": 785,
  "tracks": [
    {
      "title": "Track title",
      "artist": "Artist",
      "album": "Album",
      "duration": "3:42",
      "is_explicit": false,
      "song_url": "https://music.apple.com/us/song/example/123",
      "album_url": "https://music.apple.com/us/album/example/456?i=123"
    }
  ]
}
```

Set `is_complete` to `false` whenever the page or export is known to be
truncated. XML/plist imports preserve unresolved playlist references in the
validation report.

## Build and validate

After registering one or more sources:

```powershell
.\.venv\Scripts\music-profile.exe build --enrich
.\.venv\Scripts\music-profile.exe validate music_profile_source.json
```

Or pass source files directly:

```powershell
.\.venv\Scripts\music-profile.exe build `
  --input ".\data\muy-fuego.json" `
  --input ".\data\replay-2024.json" `
  --enrich `
  --output-dir ".\exports\profile"
```

`--enrich` sends only Apple catalog IDs to Apple's public iTunes lookup endpoint.
It does not send playlist membership or manual preferences. Omit it for a fully
offline build.

Edit `manual_preferences.json` to record favorite or disliked albums and notes.

Validation checks unique IDs, playlist-to-track references, known album
references, declared versus observed counts, canonical duplicates, and summary
statistics. Malformed and unresolved entities are reported rather than dropped.

## Replay playlists

Add every visible Replay URL with `add-playlist`; catalog (`pl.rp-…`) and
private-library (`p.…`) forms are both supported, and the same pagination and
completeness checks apply. URLs detected during local setup can be stored in the
ignored `.music-profile/detected-playlist-urls.txt` file without publishing
personal playlist identifiers.

## Privacy and limitations

- No analytics, telemetry, lyrics collection, or streaming-service writes.
- No credential scraping or private endpoints.
- No source music files are changed.
- Public catalog lookup may omit uploads, removed releases, detailed credits,
  editorial notes, or region-specific metadata.
- Library IDs and catalog IDs are different; deterministic fallback IDs prevent
  missing catalog equivalencies from collapsing unrelated tracks.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```
