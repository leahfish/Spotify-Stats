"""
Pull every song in my Spotify Liked Songs and save it to liked_songs_raw.csv.

How it works:
- SpotifyOAuth reads my app credentials from .env and logs me in. The login
  token is cached in .cache so I only have to approve access in the browser once.
- Spotify returns liked songs in pages of up to 50, so we loop and move the
  offset forward until a page comes back empty.

Note on the February 2026 API changes: Spotify removed the track "popularity"
field for Development Mode apps. We use .get() so the script doesn't crash, and
the popularity column will be empty. Popularity comes from Last.fm later instead.
"""

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()  # loads SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-library-read"))

rows, offset = [], 0
while True:
    batch = sp.current_user_saved_tracks(limit=50, offset=offset)
    if not batch["items"]:
        break

    for it in batch["items"]:
        t = it["track"]
        if t is None:  # occasionally Spotify returns an empty item; skip it
            continue
        rows.append({
            "track_id": t["id"],                       # None for local files
            "track": t["name"],
            "artist": t["artists"][0]["name"],         # first-listed (main) artist
            "artist_id": t["artists"][0]["id"],
            "n_artists": len(t["artists"]),
            "all_artists": "|".join(a["name"] for a in t["artists"]),
            "album": t["album"]["name"],
            "album_id": t["album"].get("id"),
            "album_type": t["album"]["album_type"],    # album, single, or compilation
            "album_total_tracks": t["album"].get("total_tracks"),
            "release_date": t["album"]["release_date"],
            "release_date_precision": t["album"].get("release_date_precision"),
            "duration_ms": t["duration_ms"],
            "explicit": t["explicit"],
            "popularity": t.get("popularity"),         # removed by Spotify in 2026, usually empty
            "is_local": t.get("is_local", False),      # songs from files on my computer
            "added_at": it["added_at"],                # when I liked it (UTC)
        })

    offset += 50
    print(f"Fetched {len(rows)} songs so far...")

df = pd.DataFrame(rows)
df.to_csv("liked_songs_raw.csv", index=False)
print(f"Saved {len(df)} songs to liked_songs_raw.csv")
