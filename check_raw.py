"""
Step 1: sanity-check liked_songs_raw.csv before building anything on top of it.

Checks: row count, missing values per column, duplicate songs, local files
(songs from my computer, which have no Spotify track_id), and the date range.
"""

import pandas as pd

df = pd.read_csv("liked_songs_raw.csv", encoding="utf-8")

print("=== Shape ===")
print(f"{len(df)} rows, {df.shape[1]} columns\n")

print("=== Missing values per column ===")
nulls = df.isna().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else "None")
print()

print("=== Local files (no Spotify track_id) ===")
local = df[df["track_id"].isna() | df["is_local"]]
print(f"{len(local)} local files")
if len(local):
    print(local[["track", "artist"]].head(10).to_string(index=False))
print()

print("=== Duplicates ===")
dup_ids = df["track_id"].dropna().duplicated().sum()
print(f"Duplicate track_ids: {dup_ids}")
# Same song can exist under different IDs (e.g. single and album versions)
dup_names = df.duplicated(subset=["track", "artist"]).sum()
print(f"Same title + artist under different IDs: {dup_names}")
if dup_names:
    ex = df[df.duplicated(subset=["track", "artist"], keep=False)]
    print(ex.sort_values(["artist", "track"])[["track", "artist", "album", "album_type"]]
          .head(10).to_string(index=False))
print()

print("=== Dates ===")
added = pd.to_datetime(df["added_at"], utc=True)
print(f"Liked between {added.min():%Y-%m-%d} and {added.max():%Y-%m-%d}")
print("Release date precision counts:")
print(df["release_date_precision"].value_counts(dropna=False).to_string())
print()

print("=== Quick column summaries ===")
print(df["album_type"].value_counts().to_string())
print(f"\nExplicit share: {df['explicit'].mean():.1%}")
print(f"Unique artists (main artist): {df['artist_id'].nunique()}")
print(f"Unique albums: {df['album_id'].nunique()}")
print(f"Songs with more than one artist: {(df['n_artists'] > 1).mean():.1%}")
print(f"Duration (minutes): median {df['duration_ms'].median() / 60000:.2f}, "
      f"min {df['duration_ms'].min() / 60000:.2f}, max {df['duration_ms'].max() / 60000:.2f}")
