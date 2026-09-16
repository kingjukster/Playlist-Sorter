/*
 * Local Apple Music web-player capture.
 *
 * Run this file's contents in the browser console while a playlist page is
 * open. It reads only the rendered page and downloads a JSON file locally.
 * It does not read cookies, tokens, storage, or network traffic.
 */
(async () => {
  "use strict";

  const clean = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const sleep = (milliseconds) =>
    new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  const absoluteUrl = (value) => {
    if (!value) return "";
    try {
      return new URL(value, window.location.href).href;
    } catch {
      return "";
    }
  };
  const firstLink = (row, pathPart) => {
    const link = row.querySelector(`a[href*="${pathPart}"]`);
    return link
      ? { text: clean(link.textContent), url: absoluteUrl(link.getAttribute("href")) }
      : { text: "", url: "" };
  };
  const linkData = (link) =>
    link
      ? { text: clean(link.textContent), url: absoluteUrl(link.getAttribute("href")) }
      : { text: "", url: "" };
  const songLink = (row) =>
    linkData(
      row.querySelector("a[href*='/song/'], a[href*='?i='], a[href*='&i=']")
    );
  const albumLink = (row) => {
    const links = [...row.querySelectorAll("a[href*='/album/']")];
    const collectionLink = links.find((link) => {
      const url = new URL(link.getAttribute("href"), window.location.href);
      return !url.searchParams.has("i");
    });
    return linkData(collectionLink || links.at(-1));
  };
  const albumText = (row) =>
    clean(row.querySelector(".songs-list__col--tertiary")?.textContent);
  const durationFrom = (row) => {
    const candidates = [...row.querySelectorAll("td, [role='cell'], span")]
      .map((element) => clean(element.textContent))
      .filter((value) => /^\d{1,2}:\d{2}(?::\d{2})?$/.test(value));
    return candidates.at(-1) || "";
  };
  const currentRows = () => {
    const primary = [...document.querySelectorAll("[data-testid='track-list-item']")];
    if (primary.length) return primary;
    return [...document.querySelectorAll("table tr, [role='row']")].filter(
      (row) => row.querySelector("a[href*='/song/']") || row.querySelector("button[aria-label^='Play ']")
    );
  };

  // Apple Music virtualizes long playlists in 100-row batches. Repeatedly move
  // its own scroll container to the bottom until both the row count and scroll
  // height stop growing, then restore the user's original position.
  const scroller =
    document.querySelector("#scrollable-page, .scrollable-page") ||
    document.scrollingElement;
  const originalScrollTop = scroller?.scrollTop || 0;
  let previousCount = -1;
  let previousHeight = -1;
  let stablePasses = 0;
  const maximumPasses = 30;
  for (let pass = 0; pass < maximumPasses && stablePasses < 4; pass += 1) {
    if (scroller) {
      scroller.scrollTop = scroller.scrollHeight;
      scroller.dispatchEvent(new Event("scroll", { bubbles: true }));
    }
    await sleep(750);
    const count = currentRows().length;
    const height = scroller?.scrollHeight || document.documentElement.scrollHeight;
    stablePasses =
      count === previousCount && height === previousHeight && count > 0
        ? stablePasses + 1
        : 0;
    previousCount = count;
    previousHeight = height;
    console.info(`[music-profile] loaded ${count} track rows`);
  }
  const loadStabilized = stablePasses >= 4;

  const rows = currentRows();
  if (!rows.length) {
    throw new Error("No playlist rows were found. Open a playlist, wait for it to load, and retry.");
  }

  const pageText = clean(document.body.innerText);
  const declaredCounts = [...pageText.matchAll(/\b([\d,]+)\s+songs?\b/gi)].map((match) =>
    Number(match[1].replaceAll(",", ""))
  );
  // Some private-library pages say "100 Songs" even after more batches load.
  const sourceTrackCount = Math.max(rows.length, ...declaredCounts, 0);
  const playlistId = decodeURIComponent(window.location.pathname.split("/").filter(Boolean).at(-1) || "");
  const heading = clean(document.querySelector("h1")?.textContent);
  const playlistName = heading || document.title.replace(/\s+-\s+Apple Music.*$/i, "").trim();

  const tracks = rows.map((row, index) => {
    const song = songLink(row);
    const artist = firstLink(row, "/artist/");
    const album = albumLink(row);
    let title = song.text;
    if (!title) {
      const playButton = row.querySelector("button[aria-label^='Play ']");
      title = clean(playButton?.getAttribute("aria-label")).replace(/^Play\s+/i, "");
      if (artist.text) {
        title = title.replace(new RegExp(`\\s+by\\s+${escapeRegExp(artist.text)}$`, "i"), "");
      }
    }
    return {
      position: index + 1,
      title,
      artist: artist.text,
      album: album.text || albumText(row),
      duration: durationFrom(row),
      is_explicit: Boolean(
        row.querySelector("[alt*='Explicit' i], [aria-label*='Explicit' i], [title*='Explicit' i]")
      ),
      song_url: song.url,
      album_url: album.url,
      source_url: song.url,
    };
  });

  const unresolvedItems = tracks
    .filter((track) => !track.title || !track.artist)
    .map((track) => ({
      type: "track_metadata",
      position: track.position,
      title: track.title,
      artist: track.artist,
    }));
  const complete =
    loadStabilized && rows.length === sourceTrackCount && unresolvedItems.length === 0;
  const payload = {
    playlist: {
      playlist_id: playlistId,
      name: playlistName,
      description: "",
      url: window.location.href,
      source_track_count: sourceTrackCount,
      is_complete: complete,
    },
    is_complete: complete,
    source_track_count: sourceTrackCount,
    captured_at: new Date().toISOString(),
    collection_method: "local Apple Music web-player DOM capture",
    unresolved_items: unresolvedItems,
    tracks,
  };

  const safeName = (playlistName || playlistId || "playlist")
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/gi, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
  const blobUrl = URL.createObjectURL(
    new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" })
  );
  const download = document.createElement("a");
  download.href = blobUrl;
  download.download = `apple-music-${safeName || "playlist"}.json`;
  download.style.display = "none";
  document.body.append(download);
  download.click();
  download.remove();
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  if (scroller) scroller.scrollTop = originalScrollTop;

  window.alert(
    `Captured ${tracks.length} of ${sourceTrackCount} tracks from ${playlistName}. ` +
      (complete ? "The capture is complete." : "The capture is marked incomplete; wait for the full list and retry.")
  );
})();
