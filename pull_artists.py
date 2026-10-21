"""
Step 2: try to get artist genres from Spotify.

What we found (October 2026):
- The batch endpoint sp.artists() (50 artists per call) returns 403 Forbidden.
  Spotify removed it for Development Mode apps in February 2026.
- The single-artist endpoint sp.artist(id) still works, but the response no
  longer includes "genres", "popularity", or "followers". Only the name, images,
  and links come back.

So this script checks a sample of artists first. If none of them have genres,
it stops instead of making ~1,400 calls that return nothing useful. Genres come
from Last.fm artist tags instead (see pull_lastfm.py).

If Spotify ever brings genres back, the script pulls them for every artist and
saves artists.csv, writing as it goes so a rerun picks up where it stopped.
"""

import os
import time

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-library-read"), retries=0)

OUT = "artists.csv"
artist_ids = pd.read_csv("liked_songs_raw.csv")["artist_id"].dropna().unique().tolist()
print(f"{len(artist_ids)} unique artists")

# 1) Test the batch endpoint once
try:
    sp.artists(artist_ids[:50])
    print("Batch endpoint works")
except spotipy.SpotifyException as e:
    print(f"Batch endpoint failed: HTTP {e.http_status}")

# 2) Check a sample of single lookups for a genres field
sample = [sp.artist(aid) for aid in artist_ids[:10]]
if not any(a.get("genres") for a in sample):
    print("No genres returned for any of 10 sample artists. "
          "Spotify no longer provides them for this app. Stopping here.")
    raise SystemExit

# 3) Genres are available: pull every artist, resuming from any earlier run
done = set(pd.read_csv(OUT)["artist_id"]) if os.path.exists(OUT) else set()
todo = [a for a in artist_ids if a not in done]
print(f"Pulling {len(todo)} artists ({len(done)} already saved)")

for i, aid in enumerate(todo, 1):
    try:
        a = sp.artist(aid)
    except spotipy.SpotifyException as e:
        if e.http_status == 429:  # rate limited: wait as long as Spotify asks
            wait = int(e.headers.get("Retry-After", 30)) if e.headers else 30
            print(f"Rate limited, waiting {wait}s")
            time.sleep(wait)
            a = sp.artist(aid)
        else:
            raise
    row = pd.DataFrame([{"artist_id": aid, "genres": "|".join(a.get("genres") or [])}])
    row.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    if i % 100 == 0:
        print(f"  {i}/{len(todo)}")
    time.sleep(0.2)

print(f"Saved {OUT}")
