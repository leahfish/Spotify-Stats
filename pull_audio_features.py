"""
Step 3: get audio features (danceability, energy, valence, tempo, etc.).

What we found (October 2026):
- Spotify's sp.audio_features() returns 403 Forbidden. Spotify blocked this
  endpoint for apps created after November 27, 2024.
- ReccoBeats (https://reccobeats.com) is a free API with no key that accepts
  Spotify track IDs and returns Spotify-style audio features. Its numbers are
  ReccoBeats' own estimates, not Spotify's originals.

ReccoBeats details we learned by testing:
- Up to 40 IDs per call (more gives a 400 error).
- Results don't come back in the order we sent them. Each result has the
  Spotify link in "href", so we pull the Spotify ID out of that.
- Songs ReccoBeats doesn't know are simply left out of the response.

Output: audio_features.csv, one row per matched song. Rows are saved after each
batch, and a rerun skips songs that were already looked up.
"""

import os
import time

import pandas as pd
import requests

URL = "https://api.reccobeats.com/v1/audio-features"
OUT = "audio_features.csv"
CHECKED = "audio_features_checked.txt"  # every ID we've asked about, matched or not
BATCH = 40
FEATURES = ["acousticness", "danceability", "energy", "instrumentalness", "key",
            "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

# Liked Songs, plus songs from my playlists once pull_playlists.py has been run
sources = ["liked_songs_raw.csv"] + (["playlist_tracks.csv"] if os.path.exists("playlist_tracks.csv") else [])
ids = pd.concat(pd.read_csv(f) for f in sources).dropna(subset=["track", "track_id"])["track_id"].unique().tolist()

# Resume support: skip IDs we've already asked about
checked = set(open(CHECKED).read().split()) if os.path.exists(CHECKED) else set()
todo = [i for i in ids if i not in checked]
print(f"{len(ids)} songs, {len(todo)} still to look up")


def get_batch(batch_ids):
    """Call ReccoBeats for up to 40 IDs, waiting and retrying if rate limited
    or if the connection drops (this happened once during the full run)."""
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"ids": ",".join(batch_ids)}, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            wait = 15 * (attempt + 1)  # wait longer after each failure
            print(f"  Connection problem, retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"  Rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json().get("content", [])
    raise RuntimeError("Still rate limited after 5 tries")


for start in range(0, len(todo), BATCH):
    batch_ids = todo[start:start + BATCH]
    results = get_batch(batch_ids)

    rows = []
    for item in results:
        spotify_id = item["href"].rstrip("/").split("/")[-1]  # .../track/<id>
        row = {"track_id": spotify_id, "reccobeats_id": item["id"], "isrc": item.get("isrc")}
        row.update({f: item.get(f) for f in FEATURES})
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    with open(CHECKED, "a") as f:
        f.write("\n".join(batch_ids) + "\n")

    done = start + len(batch_ids)
    if done % 400 == 0 or done == len(todo):
        print(f"  {done}/{len(todo)} looked up")
    time.sleep(0.5)  # be polite to a free API

# Summary
af = pd.read_csv(OUT).drop_duplicates("track_id")
print(f"\nMatched {len(af)} of {len(ids)} songs ({len(af) / len(ids):.1%})")
print(af[FEATURES].describe().round(3).to_string())
