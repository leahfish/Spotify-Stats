"""
Pull every song in every playlist I made and save it to playlist_tracks.csv.

What we learned by testing (test_playlists.py, October 2026):
- Spotify only lets this app read songs from playlists I own. Playlists made by
  other people return 403 Forbidden, so they're skipped.
- The old endpoint playlists/{id}/tracks returns 403 for Development Mode apps.
  The newer playlists/{id}/items works, so we call it directly. (spotipy's
  playlist_items() still uses the old one.)

One row per song per playlist, so a song in 3 playlists appears 3 times. The raw
data is kept as-is here; merging playlists (e.g. my three "Country" playlists)
happens later in the model notebook.

Usage:
    python pull_playlists.py          # all my playlists
    python pull_playlists.py --test   # first 2 playlists, print only, save nothing
"""

import sys

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

sys.stdout.reconfigure(encoding="utf-8")  # playlist names can contain emoji
load_dotenv()
SCOPES = "user-library-read playlist-read-private playlist-read-collaborative"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))
TEST = "--test" in sys.argv

# Step 1: list my own playlists (50 per page)
me = sp.current_user()["id"]
playlists, offset = [], 0
while True:
    page = sp.current_user_playlists(limit=50, offset=offset)
    playlists += [p for p in page["items"] if p and p["owner"]["id"] == me]
    if not page["next"]:
        break
    offset += 50
# The same playlist can appear twice in my library list (Summer19 did), so keep each ID once
playlists = list({p["id"]: p for p in playlists}.values())
print(f"{len(playlists)} playlists I made")
if TEST:
    playlists = playlists[:2]

# Step 2: page through each playlist's songs (50 per call)
rows = []
for p in playlists:
    offset = 0
    while True:
        page = sp._get(f"playlists/{p['id']}/items", limit=50, offset=offset)
        for it in page["items"]:
            t = it.get("item") or it.get("track")  # newer responses call it "item"
            if not t or t.get("type") != "track":  # skip podcast episodes and empty entries
                continue
            rows.append({
                "playlist_id": p["id"],
                "playlist": p["name"].strip(),  # some names end in stray spaces ("Sweet  ")
                "added_to_playlist": it.get("added_at"),
                "track_id": t["id"],  # None for local files
                "track": t["name"],
                "artist": t["artists"][0]["name"] if t["artists"] else None,
                "artist_id": t["artists"][0]["id"] if t["artists"] else None,
                "n_artists": len(t["artists"]),
                "all_artists": "|".join(a["name"] for a in t["artists"]),
                "album": t["album"]["name"],
                "album_id": t["album"]["id"],
                "album_type": t["album"]["album_type"],
                "release_date": t["album"]["release_date"],
                "release_date_precision": t["album"]["release_date_precision"],
                "duration_ms": t["duration_ms"],
                "explicit": t["explicit"],
                "is_local": t.get("is_local", False),
            })
        if not page["next"]:
            break
        offset += 50
    print(f"  {p['name'][:40]:40} {sum(r['playlist_id'] == p['id'] for r in rows)} songs")

df = pd.DataFrame(rows)
if TEST:
    print(df.head(5).T.to_string())
    raise SystemExit

df.to_csv("playlist_tracks.csv", index=False)
liked = set(pd.read_csv("liked_songs_raw.csv")["track_id"].dropna())
print(f"\nSaved playlist_tracks.csv: {len(df)} rows, {df['track_id'].nunique()} unique songs")
print(f"Unique songs also in my Liked Songs: {df['track_id'].dropna().drop_duplicates().isin(liked).mean():.1%}")
print(f"Local files (no Spotify ID): {df['track_id'].isna().sum()}")
