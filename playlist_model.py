"""
Shared pieces of the playlist model, used by playlists.ipynb (training) and
predict_playlist.py (predicting for any song).

They live in a regular .py file because a saved model can only be loaded again
if the custom pieces inside it (like split_tags) can be imported by name.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

SEED = 42

# My vibe playlists and the label each one counts toward
LABELS = {
    "Country": "Country",
    "Up There": "Hip Hop/Rap", "Jump Around": "Hip Hop/Rap",
    "Giggles": "Cute", "Sweet": "Cute",
    "Aspiring Rory Gilmore": "Fall", "pumpkin playlist": "Fall",
    "Boat": "Boat Songs",
    "paradise": "Hype", "Running": "Hype", "breezy": "Hype",
    "Sad.": "Sad Vibes", "Sad pop princesses": "Sad Vibes", "snowy": "Sad Vibes", "Dive": "Sad Vibes",
    "pink skies": "Sunset",
}

AUDIO = ["danceability", "energy", "valence", "acousticness", "speechiness",
         "instrumentalness", "liveness", "tempo", "loudness", "mode"]
# Features that describe me rather than the song (whether I liked it, how much I've played it)
PERSONAL = ["in_liked", "my_plays"]
BASIC = AUDIO + ["release_year", "duration_min", "explicit", "n_artists", "is_single",
                 "log_track_listeners", "log_artist_listeners"] + PERSONAL
SONG_ONLY = [c for c in BASIC if c not in PERSONAL]


def build_features(songs, audio, lfm_tracks, lfm_artists, liked_ids):
    """One feature row per song.

    songs: indexed by track_id, with artist_id, n_artists, album_type, release_date, duration_ms, explicit
    audio: ReccoBeats features indexed by track_id (songs missing from it get blanks)
    lfm_tracks / lfm_artists: Last.fm results indexed by track_id / artist_id
    liked_ids: track IDs in my Liked Songs
    """
    feat = audio.reindex(index=songs.index, columns=AUDIO).astype(float)
    feat["release_year"] = pd.to_datetime(songs["release_date"], format="mixed", errors="coerce").dt.year
    feat["duration_min"] = songs["duration_ms"] / 60000
    feat["explicit"] = songs["explicit"].astype(int)
    feat["n_artists"] = songs["n_artists"]
    feat["is_single"] = (songs["album_type"] == "single").astype(int)
    feat["log_track_listeners"] = np.log10(lfm_tracks["lfm_track_listeners"].reindex(songs.index) + 1)
    feat["log_artist_listeners"] = np.log10(lfm_artists["lfm_artist_listeners"].reindex(songs["artist_id"]).values + 1)
    feat["in_liked"] = songs.index.isin(liked_ids).astype(int)
    feat["my_plays"] = lfm_tracks["lfm_user_playcount"].reindex(songs.index)

    # Tags: the song's own Last.fm tags, or its artist's if it has none. "No tags" can arrive as an empty
    # string (straight from the API) or a blank (read back from a CSV), so both count as missing.
    track_tags = lfm_tracks["lfm_track_tags"].reindex(songs.index).replace("", np.nan)
    artist_tags = pd.Series(lfm_artists["lfm_artist_tags"].reindex(songs["artist_id"]).values,
                            index=songs.index).replace("", np.nan)
    feat["tags"] = track_tags.fillna(artist_tags).fillna("").str.replace("hip hop", "hip-hop")
    feat["artist_id"] = songs["artist_id"]
    return feat


VERSION_WORDS = r"remaster\w*|version|edit|live|mono|stereo|from|sessions?|soundtrack|bonus|deluxe|single|recorded|demo"


def song_key(title, artist):
    """Lowercase title + primary artist with feat. credits, version suffixes, and punctuation removed,
    so different versions of the same song (remaster, live, Taylor's Version) share one key."""
    t = str(title).lower().replace("’", "'")
    t = re.sub(r"\s*[\(\[](feat\.?|ft\.?|featuring|with)\s[^\)\]]*[\)\]]", "", t)  # (feat. X), [with Y]
    t = re.sub(rf"\s+-\s+.*\b({VERSION_WORDS})\b.*$", "", t)                       # - 2011 Remaster
    t = re.sub(rf"\s*[\(\[][^\)\]]*\b({VERSION_WORDS})\b[^\)\]]*[\)\]]", "", t)    # (Taylor's Version)
    t = " ".join(re.sub(r"[^a-z0-9 ]", "", t).split())
    a = " ".join(re.sub(r"[^a-z0-9 ]", "", str(artist).lower()).split())
    return f"{t} | {a}"


def split_tags(s):
    return [t for t in s.split("|") if t]


class ArtistHistory(BaseEstimator, TransformerMixin):
    """Share of each artist's training songs in each label, smoothed toward the overall share, plus the
    artist's song count. fit_transform (training songs) uses an inner K-fold so a song never sees its own label."""

    def __init__(self, smooth=2, n_splits=5):
        self.smooth = smooth
        self.n_splits = n_splits

    def _shares(self, known_artists, known_Y, apply_artists):
        sums = pd.DataFrame(known_Y).groupby(known_artists).sum()
        counts = pd.Series(known_artists).value_counts()
        s = sums.reindex(apply_artists).fillna(0).values
        n = counts.reindex(apply_artists).fillna(0).values[:, None]
        return np.hstack([(s + self.smooth * known_Y.mean(axis=0)) / (n + self.smooth), n])

    def fit(self, X, y):
        self.artists_ = np.asarray(X).ravel()
        self.Y_ = np.asarray(y, dtype=float)
        return self

    def transform(self, X):
        return self._shares(self.artists_, self.Y_, np.asarray(X).ravel())

    def fit_transform(self, X, y):
        self.fit(X, y)
        out = np.zeros((len(self.artists_), self.Y_.shape[1] + 1))
        for fit_i, apply_i in KFold(self.n_splits, shuffle=True, random_state=0).split(self.artists_):
            out[apply_i] = self._shares(self.artists_[fit_i], self.Y_[fit_i], self.artists_[apply_i])
        return out


class Baseline(BaseEstimator):
    """Always predicts each label's share of the training songs."""

    def fit(self, X, y):
        self.shares_ = np.asarray(y).mean(axis=0)
        return self

    def predict_proba(self, X):
        return np.tile(self.shares_, (len(X), 1))


def make_version(use_tags, use_artist, basic=BASIC):
    """The model pipeline: chosen basic columns, optional tag columns and artist history, then
    one gradient-boosted tree model per label."""
    parts = [("basic", "passthrough", basic)]
    if use_tags:
        parts.append(("tags", CountVectorizer(tokenizer=split_tags, token_pattern=None, lowercase=False,
                                              binary=True, min_df=15), "tags"))
    if use_artist:
        parts.append(("artist", ArtistHistory(), ["artist_id"]))
    model = OneVsRestClassifier(HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED))
    return Pipeline([("features", ColumnTransformer(parts, sparse_threshold=0)), ("model", model)])


def top_k(Y_true, probs):
    """Share of songs whose first guess is a real label (top-1), and with a real label in the top 3 (top-3)."""
    rank = np.argsort(-probs, axis=1)
    truth = np.asarray(Y_true)
    top1 = truth[np.arange(len(truth)), rank[:, 0]].mean()
    top3 = (np.take_along_axis(truth, rank[:, :3], axis=1).sum(axis=1) > 0).mean()
    return top1, top3
