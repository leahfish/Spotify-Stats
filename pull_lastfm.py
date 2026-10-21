"""
Step 3b: get popularity and genre tags from Last.fm.

Why Last.fm: Spotify removed track popularity, artist popularity, follower
counts, and genres for Development Mode apps in 2026. Last.fm is free (with an
API key) and tracks how many people listen to each song and artist, plus
user-submitted tags like "indie pop" or "chill" that work as genres and moods.

What we pull:
- track.getInfo  -> listeners, playcount, top tags, and my own play count
- artist.getInfo -> listeners, playcount, tags, and my own play count for each main artist

Matching: Last.fm doesn't use Spotify IDs, so songs are matched on artist name
+ track title. We try the title exactly as Spotify has it first. If that fails,
we try a cleaned title (feat. credits, "- 2010 Remaster" style suffixes, and
curly quotes removed). The lfm_match column records which one worked.
autocorrect=1 also lets Last.fm fix small spelling differences.

Things we deliberately keep in the title: "Remix" and "(Taylor's Version)",
because those are different recordings with their own Last.fm stats.

The key is read from .env as LASTFM_API_KEY and never printed.
Results are saved as we go, and a rerun skips anything already looked up.
Last.fm asks for no more than about 5 requests per second, so we pause between calls.

Usage:
    python pull_lastfm.py          # full run (artists, then tracks)
    python pull_lastfm.py --test   # try 20 songs and 5 artists, save nothing
"""

import os
import re
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LASTFM_API_KEY")
if not API_KEY:
    raise SystemExit("LASTFM_API_KEY is missing from .env")

API = "https://ws.audioscrobbler.com/2.0/"
LASTFM_USER = "leahfishy"  # my Last.fm username, used to get my own play counts
PAUSE = 0.25  # seconds between calls, about 4 per second
TEST = "--test" in sys.argv


# Spotify names that Last.fm spells differently
ARTIST_FIXES = {"Axwell /\\ Ingrosso": "Axwell Λ Ingrosso"}  # Last.fm uses the Greek letter Λ


def call(method, **params):
    """Make one Last.fm request, retrying on connection problems or rate limits."""
    if "artist" in params:
        params["artist"] = ARTIST_FIXES.get(params["artist"], params["artist"])
    # Last.fm decodes names twice, so a normal "+" turns into a space ("Dan + Shay" -> "Dan   Shay",
    # which is a different, nearly empty page). Pre-encoding it as %2B makes it arrive as "+".
    for key in ("artist", "track"):
        if key in params:
            params[key] = params[key].replace("+", "%2B")
    params.update({"method": method, "api_key": API_KEY, "format": "json",
                   "autocorrect": 1, "username": LASTFM_USER})
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, timeout=20)
            data = r.json()
        except (requests.ConnectionError, requests.Timeout, ValueError):
            time.sleep(10 * (attempt + 1))
            continue
        # Error 29 = rate limit exceeded, 8/11/16 = temporary server problems
        if data.get("error") in (8, 11, 16, 29):
            time.sleep(10 * (attempt + 1))
            continue
        return data  # either a result or a "not found" error (code 6)
    return {"error": "gave_up"}


# Words that mark a version suffix we can safely drop, like "- 2010 Remaster"
# or '- From "Frozen"'. "remix" is intentionally not on this list.
SUFFIX_WORDS = r"remaster\w*|version|edit|live|mono|stereo|from|sessions?|soundtrack|bonus|deluxe|single|recorded|demo"


def clean_title(title):
    """Return a simplified title for a second matching attempt."""
    t = title.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # "(feat. Drake)", "[with SZA]", "(ft. X & Y)"
    t = re.sub(r"\s*[\(\[](feat\.?|ft\.?|featuring|with)\s[^\)\]]*[\)\]]", "", t, flags=re.I)
    # " - 2010 Remaster", ' - From "Pitch Perfect 2" Soundtrack' (but not Taylor's Version)
    if "taylor's version" not in t.lower():
        t = re.sub(rf"\s+-\s+.*\b({SUFFIX_WORDS})\b.*$", "", t, flags=re.I)
        # "(Remastered 2009)", "(Single Version)", "(Mono)"
        t = re.sub(rf"\s*[\(\[][^\)\]]*\b({SUFFIX_WORDS})\b[^\)\]]*[\)\]]", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def tag_list(tags_obj):
    """Turn Last.fm's tag structure into 'tag1|tag2|...' (top 5)."""
    tags = (tags_obj or {}).get("tag", [])
    if isinstance(tags, dict):  # a single tag comes back as a dict, not a list
        tags = [tags]
    return "|".join(t["name"].lower() for t in tags[:5])


def track_row(artist, track, track_id):
    """Look up one song: original title first, then the cleaned title."""
    data = call("track.getInfo", artist=artist, track=track)
    match = "original"
    cleaned = clean_title(track)
    if "track" not in data and cleaned != track:
        time.sleep(PAUSE)
        data = call("track.getInfo", artist=artist, track=cleaned)
        match = "cleaned"
    t = data.get("track")
    if not t:
        return {"track_id": track_id, "lfm_match": "none"}
    return {
        "track_id": track_id,
        "lfm_match": match,
        "lfm_track_name": t.get("name"),
        "lfm_artist_name": t.get("artist", {}).get("name"),
        "lfm_track_listeners": int(t.get("listeners", 0)),
        "lfm_track_playcount": int(t.get("playcount", 0)),
        "lfm_user_playcount": int(t.get("userplaycount", 0)),
        "lfm_user_loved": t.get("userloved") == "1",
        "lfm_track_tags": tag_list(t.get("toptags")),
    }


def artist_row(artist, artist_id):
    data = call("artist.getInfo", artist=artist)
    a = data.get("artist")
    if not a:
        return {"artist_id": artist_id, "lfm_artist_matched": False}
    stats = a.get("stats", {})
    return {
        "artist_id": artist_id,
        "lfm_artist_matched": True,
        "lfm_artist_listeners": int(stats.get("listeners", 0)),
        "lfm_artist_playcount": int(stats.get("playcount", 0)),
        "lfm_artist_user_playcount": int(stats.get("userplaycount", 0)),
        "lfm_artist_tags": tag_list(a.get("tags")),
    }


def run(items, make_row, out_file, id_col, label):
    """Look up each item, appending rows to out_file and skipping ones already saved."""
    done = set(pd.read_csv(out_file)[id_col]) if os.path.exists(out_file) else set()
    todo = [it for it in items if it[-1] not in done]
    print(f"{label}: {len(todo)} to look up ({len(done)} already saved)")
    batch = []
    for i, it in enumerate(todo, 1):
        batch.append(make_row(*it))
        time.sleep(PAUSE)
        if len(batch) == 50 or i == len(todo):  # save every 50 lookups
            pd.DataFrame(batch).to_csv(out_file, mode="a", header=not os.path.exists(out_file), index=False)
            batch = []
        if i % 500 == 0:
            print(f"  {i}/{len(todo)}")


def main():
    # Liked Songs, plus songs from my playlists once pull_playlists.py has been run
    sources = ["liked_songs_raw.csv"] + (["playlist_tracks.csv"] if os.path.exists("playlist_tracks.csv") else [])
    df = pd.concat(pd.read_csv(f) for f in sources).dropna(subset=["track", "track_id"])
    artists = df.drop_duplicates("artist_id")[["artist", "artist_id"]].values.tolist()
    tracks = df.drop_duplicates("track_id")[["artist", "track", "track_id"]].values.tolist()

    if TEST:
        # 12 random songs plus 8 whose titles the cleaner changes, so both paths get tested
        needs_clean = df[df["track"].map(clean_title) != df["track"]]
        sample = pd.concat([df.sample(12, random_state=7), needs_clean.sample(8, random_state=7)])
        rows = [track_row(a, t, tid) | {"spotify_title": t}
                for a, t, tid in sample[["artist", "track", "track_id"]].values]
        cols = ["spotify_title", "lfm_match", "lfm_track_listeners", "lfm_user_playcount", "lfm_track_tags"]
        pd.set_option("display.width", 200, "display.max_colwidth", 45)
        print(pd.DataFrame(rows)[cols].to_string(index=False))
        print()
        print(pd.DataFrame([artist_row(a, aid) for a, aid in artists[:5]]).drop(columns="artist_id").to_string(index=False))
        return

    run(artists, artist_row, "lastfm_artists.csv", "artist_id", "Artists")
    run(tracks, track_row, "lastfm_tracks.csv", "track_id", "Tracks")

    # Summary, plus a list of songs Last.fm couldn't find
    la = pd.read_csv("lastfm_artists.csv").drop_duplicates("artist_id")
    lt = pd.read_csv("lastfm_tracks.csv").drop_duplicates("track_id")
    print(f"\nArtists matched: {la['lfm_artist_matched'].sum()} of {len(la)} "
          f"({la['lfm_artist_matched'].mean():.1%}), "
          f"with tags: {la['lfm_artist_tags'].notna().sum()}")
    print(f"Tracks: {len(lt)}, match type:")
    print(lt["lfm_match"].value_counts().to_string())
    print(f"Matched tracks with tags: {lt['lfm_track_tags'].notna().sum()}")
    unmatched = df[df["track_id"].isin(lt.loc[lt["lfm_match"] == "none", "track_id"])]
    unmatched[["track_id", "track", "artist"]].to_csv("lastfm_unmatched.csv", index=False)
    print(f"Unmatched songs saved to lastfm_unmatched.csv ({len(unmatched)})")


# Only pull when run as a script, so predict_playlist.py can import track_row and artist_row
if __name__ == "__main__":
    main()
