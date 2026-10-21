"""
Test: can this app read my playlists and the songs inside them?

This needs two extra permissions ("scopes") beyond Liked Songs:
- playlist-read-private: my private playlists
- playlist-read-collaborative: collaborative playlists I'm part of

Because the scopes changed, Spotify will ask me to approve access in the
browser once. The new login is saved in .cache like before (user-library-read
is kept, so pull_songs.py still works).

Nothing is saved to a file here. It only prints what it finds.
"""

import sys

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

# Playlist names can contain emoji, which the Windows terminal can't print by default
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
SCOPES = "user-library-read playlist-read-private playlist-read-collaborative"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))

me = sp.current_user()["id"]

# Step 1: list every playlist in my library, 50 at a time
playlists, offset = [], 0
while True:
    page = sp.current_user_playlists(limit=50, offset=offset)
    playlists += [p for p in page["items"] if p]  # skip empty entries
    if not page["next"]:
        break
    offset += 50


def size(p):
    # Track count has moved between field names in Spotify's API, so check both
    return (p.get("tracks") or p.get("items") or {}).get("total")


owned = [p for p in playlists if p["owner"]["id"] == me]
followed = [p for p in playlists if p["owner"]["id"] != me]
print(f"{len(playlists)} playlists: {len(owned)} made by me, {len(followed)} made by others\n")
print("My playlists (name, songs):")
for p in owned:
    print(f"  {p['name'][:45]:45} {size(p)}")


# Step 2: try reading the first few songs of one playlist of each kind
def try_read(p, label):
    print(f"\n{label}: '{p['name']}'")
    for path in [f"playlists/{p['id']}/tracks", f"playlists/{p['id']}/items"]:
        try:
            data = sp._get(path, limit=3)
            names = [(it.get("track") or it.get("item") or {}).get("name") for it in data["items"]]
            print(f"  {path.split('/')[-1]}: OK, first songs: {names}")
            return
        except spotipy.SpotifyException as e:
            print(f"  {path.split('/')[-1]}: failed ({e.http_status})")


if owned:
    try_read(owned[0], "Playlist I made")
if followed:
    try_read(followed[0], "Playlist made by someone else")
