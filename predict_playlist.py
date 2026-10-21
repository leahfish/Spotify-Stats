"""
Predict which of my vibe playlists a song belongs in. Works for any song on Spotify.

Usage:
    python predict_playlist.py "Pink Pony Club" "Chappell Roan"
    python predict_playlist.py "Pink Pony Club"            # title only (Spotify picks the top match)
    python predict_playlist.py https://open.spotify.com/track/1k2pQc5i348DCHwbn5KTdc

What it does, for one song:
1. Finds the song on Spotify (no login needed, just the app keys in .env).
2. Gets its audio features from ReccoBeats (some songs aren't there; the model handles blanks).
3. Gets listeners and tags from Last.fm, using the same matching as pull_lastfm.py.
4. Builds the same feature row the model was trained on (build_features in playlist_model.py).
5. Ranks my 8 vibe playlists with the saved model, models/playlist_model.joblib.

The saved model is the "song only" version from playlists.ipynb section 8: it leaves out whether a
song is in my Liked Songs and my play counts, since a new song never has those. That version was
checked with cross-validation only, not on the final test set.
"""

import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from playlist_model import build_features
from pull_lastfm import artist_row, track_row  # same Last.fm matching ("+" fix, cleaned titles) as the full pull

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

if len(sys.argv) < 2:
    raise SystemExit('Usage: python predict_playlist.py "Song title" "Artist"   (or a Spotify track link)')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# Step 1: find the song on Spotify
link = re.search(r"track/([A-Za-z0-9]{22})", sys.argv[1])
if link:
    t = sp.track(link.group(1))
else:
    query = f"track:{sys.argv[1]}" + (f" artist:{sys.argv[2]}" if len(sys.argv) > 2 else "")
    found = sp.search(q=query, type="track", limit=1)["tracks"]["items"]
    if not found:
        raise SystemExit(f"Spotify found no song for: {query}")
    t = found[0]

tid, artist, artist_id = t["id"], t["artists"][0]["name"], t["artists"][0]["id"]
songs = pd.DataFrame([{
    "track_id": tid, "artist_id": artist_id, "n_artists": len(t["artists"]),
    "album_type": t["album"]["album_type"], "release_date": t["album"]["release_date"],
    "duration_ms": t["duration_ms"], "explicit": t["explicit"],
}]).set_index("track_id")

# Step 2: audio features from ReccoBeats (results say which song they belong to in "href")
r = requests.get("https://api.reccobeats.com/v1/audio-features", params={"ids": tid}, timeout=20)
matches = [row for row in r.json().get("content", []) if tid in row.get("href", "")]
audio = pd.DataFrame(matches[:1], index=[tid] * len(matches[:1]))  # empty if ReccoBeats doesn't know the song

# Step 3: Last.fm listeners and tags for the song and its artist
lfm_track = pd.DataFrame([track_row(artist, t["name"], tid)]).set_index("track_id").reindex(
    columns=["lfm_match", "lfm_track_listeners", "lfm_user_playcount", "lfm_track_tags"])
lfm_artist = pd.DataFrame([artist_row(artist, artist_id)]).set_index("artist_id").reindex(
    columns=["lfm_artist_listeners", "lfm_artist_tags"])

# Step 4 and 5: same features as training, then rank the playlists
feat = build_features(songs, audio, lfm_track, lfm_artist, liked_ids=[])
bundle = joblib.load(Path(__file__).parent / "models" / "playlist_model.joblib")
probs = pd.Series(bundle["model"].predict_proba(feat)[0], index=bundle["labels"]).sort_values(ascending=False)

# What the prediction is based on
own_tags = lfm_track["lfm_track_tags"].iloc[0]
tags = feat["tags"].iloc[0]
print(f"\n{t['name']} - {', '.join(a['name'] for a in t['artists'])} ({t['album']['release_date'][:4]})")
print(f"  Audio features: {'found' if len(audio) else 'not in ReccoBeats (prediction leans on tags and basics)'}")
listeners = lfm_track["lfm_track_listeners"].iloc[0]
print(f"  Last.fm: {f'{int(listeners):,} listeners' if pd.notna(listeners) else 'no match'}")
source = "the song" if isinstance(own_tags, str) and own_tags else "the artist" if tags else "none found"
print(f"  Tags (from {source}): {tags.replace('|', ', ') if tags else '-'}")

print("\nBest fits (each playlist is scored on its own, so these don't add up to 100%):")
for rank, (label, p) in enumerate(probs.items(), 1):
    print(f"  {rank}. {label:12} {p:5.0%}  {'#' * round(p * 30)}")

# If the song is already in one of my playlists, show where, as a check
if Path("playlist_tracks.csv").exists():
    mine = pd.read_csv("playlist_tracks.csv")
    already = sorted(mine.loc[mine["track_id"] == tid, "playlist"].unique())
    if already:
        print(f"\n(Already in my playlists: {', '.join(already)})")
