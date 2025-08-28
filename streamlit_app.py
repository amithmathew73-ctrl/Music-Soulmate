"""
Music Soulmate Engine — Streamlit App

A taste-driven music discovery tool that takes one (or more) Spotify playlists,
extracts their audio "DNA", clusters your moods, and generates deeply compatible
song recommendations across eras and languages. You can export results or create
new Spotify playlists right from the app.

──────────────────────────────────────────────────────────────────────────────
Quick Start
1) Create a Spotify Developer app at https://developer.spotify.com/
   - Add Redirect URI: http://localhost:8501
   - Copy Client ID and Client Secret
2) Install requirements (create a virtualenv if you like):
     pip install -r requirements.txt
   If you don't have a requirements.txt yet, use this list:
     streamlit==1.37.1
     spotipy==2.24.0
     scikit-learn==1.5.1
     pandas==2.2.2
     numpy==1.26.4
     python-slugify==8.0.4
3) Set environment variables (recommended) or paste in the UI fields:
     export SPOTIPY_CLIENT_ID="your_client_id"
     export SPOTIPY_CLIENT_SECRET="your_client_secret"
     export SPOTIPY_REDIRECT_URI="http://localhost:8501"
4) Run the app:
     streamlit run streamlit_app.py

──────────────────────────────────────────────────────────────────────────────
Notes
- Scopes requested: playlist-read-private, user-library-read,
  playlist-modify-public, playlist-modify-private
- Handles large playlists with pagination; caches audio features.
- Multilingual friendly via genre-language heuristics and era filters.
- You own your tokens; nothing is sent anywhere except Spotify's API.
"""

import os
import time
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st
from slugify import slugify

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --------------------------- UI CONFIG -------------------------------------
st.set_page_config(
    page_title="Music Soulmate Engine",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Music Soulmate Engine")
st.caption(
    "Taste-first recommendations across eras and languages. Bring a playlist; leave with gems."
)

# --------------------------- AUTH & SESSION --------------------------------
DEFAULT_SCOPES = (
    "playlist-read-private user-library-read "
    "playlist-modify-public playlist-modify-private"
)

with st.sidebar:
    st.header("Spotify Authentication")
    client_id = st.text_input(
        "Client ID",
        value=os.getenv("SPOTIPY_CLIENT_ID", ""),
        type="password",
        help="Stored only in your local session.",
    )
    client_secret = st.text_input(
        "Client Secret",
        value=os.getenv("SPOTIPY_CLIENT_SECRET", ""),
        type="password",
    )
    redirect_uri = st.text_input(
        "Redirect URI",
        value=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8501"),
    )
    scope = st.text_area("Scopes", value=DEFAULT_SCOPES, height=80)
    auth_btn = st.button("Connect to Spotify", type="primary")

@st.cache_resource(show_spinner=False)
def get_spotify(client_id: str, client_secret: str, redirect_uri: str, scope: str):
    auth_mgr = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        open_browser=True,
        cache_path=".spotipy_cache",
    )
    sp = spotipy.Spotify(auth_manager=auth_mgr)
    return sp

if "sp" not in st.session_state:
    st.session_state.sp = None

if auth_btn:
    try:
        st.session_state.sp = get_spotify(client_id, client_secret, redirect_uri, scope)
        me = st.session_state.sp.me()
        st.success(f"Authenticated as {me['display_name']} ({me['id']})")
    except Exception as e:
        st.error(f"Auth failed: {e}")

sp = st.session_state.sp

# --------------------------- HELPERS ---------------------------------------
LANG_GENRE_HINTS: Dict[str, List[str]] = {
    "tamil": ["tamil"],
    "malayalam": ["malayalam"],
    "hindi": ["bollywood", "desi", "hindustani", "indian pop", "punjabi", "indian indie"],
    "spanish": ["latin", "spanish", "cancion", "mexican", "argentino", "urbano latino", "reggaeton"],
    "english": ["indie", "rock", "pop", "singer-songwriter", "uk", "us", "british", "american", "folk", "alt rock", "alt pop"],
}

AUDIO_FEATURE_COLUMNS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    "tempo",
]

@st.cache_data(show_spinner=False)
def get_playlist_tracks(sp: spotipy.Spotify, playlist_input: str) -> pd.DataFrame:
    """Fetch all tracks from a playlist URL or ID, with pagination."""
    try:
        # Accept URL or raw ID
        if playlist_input.startswith("http"):
            playlist_id = playlist_input.split("playlist/")[-1].split("?")[0]
        else:
            playlist_id = playlist_input

        results = sp.playlist_items(playlist_id, additional_types=["track"], limit=100)
        items = results.get("items", [])
        while results.get("next"):
            results = sp.next(results)
            items.extend(results.get("items", []))

        records = []
        for it in items:
            tr = it.get("track")
            if not tr or tr.get("id") is None:
                continue
            artists = ", ".join([a["name"] for a in tr.get("artists", [])])
            artist_ids = [a["id"] for a in tr.get("artists", []) if a.get("id")]
            records.append({
                "track_id": tr["id"],
                "track_name": tr["name"],
                "artist_name": artists,
                "artist_ids": artist_ids,
                "album": tr.get("album", {}).get("name"),
                "release_date": tr.get("album", {}).get("release_date"),
                "duration_ms": tr.get("duration_ms"),
                "popularity": tr.get("popularity"),
                "preview_url": tr.get("preview_url"),
            })
        df = pd.DataFrame.from_records(records)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch playlist: {e}")

@st.cache_data(show_spinner=False)
def get_audio_features(sp: spotipy.Spotify, track_ids: List[str]) -> pd.DataFrame:
    feats = []
    batch = 100
    for i in range(0, len(track_ids), batch):
        chunk = track_ids[i:i+batch]
        features = sp.audio_features(chunk)
        for f in features:
            if f is None:
                continue
            row = {k: f.get(k) for k in AUDIO_FEATURE_COLUMNS + ["id", "key", "mode", "time_signature"]}
            feats.append(row)
    return pd.DataFrame(feats).rename(columns={"id": "track_id"})

@st.cache_data(show_spinner=False)
def get_artist_genres(sp: spotipy.Spotify, artist_ids: List[str]) -> Dict[str, List[str]]:
    genres: Dict[str, List[str]] = {}
    batch = 50
    for i in range(0, len(artist_ids), batch):
        chunk = artist_ids[i:i+batch]
        artists = sp.artists(chunk)["artists"]
        for a in artists:
            if not a:
                continue
            genres[a["id"]] = a.get("genres", [])
    return genres

def infer_languages(artist_genre_map: Dict[str, List[str]]) -> Set[str]:
    langs: Set[str] = set()
    for gid, glist in artist_genre_map.items():
        gstr = " ".join(glist).lower()
        for lang, hints in LANG_GENRE_HINTS.items():
            if any(h in gstr for h in hints):
                langs.add(lang)
    return langs

def choose_k(df_features: pd.DataFrame, k_min: int = 3, k_max: int = 8) -> int:
    X = df_features[AUDIO_FEATURE_COLUMNS].copy()
    X["loudness"] = X["loudness"].fillna(X["loudness"].median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    best_k, best_score = k_min, -1
    for k in range(k_min, min(k_max, len(Xs)) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(Xs, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k

def build_clusters(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    X = df[AUDIO_FEATURE_COLUMNS].copy()
    X["loudness"] = X["loudness"].fillna(X["loudness"].median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    labels = km.fit_predict(Xs)
    dfc = df.copy()
    dfc["cluster"] = labels
    return dfc, km, scaler

def centroid_targets(centroid: np.ndarray, scaler: StandardScaler) -> Dict[str, float]:
    inv = scaler.inverse_transform(centroid.reshape(1, -1))[0]
    target = dict(zip(AUDIO_FEATURE_COLUMNS, inv))
    # Spotify expects positive targets; clip where sensible
    target["loudness"] = float(target["loudness"])  # can be negative, OK
    target["tempo"] = max(40.0, min(220.0, float(target["tempo"])))
    # The rest in [0,1]
    for k in ["acousticness","danceability","energy","instrumentalness","liveness","speechiness","valence"]:
        target[k] = float(np.clip(target[k], 0.0, 1.0))
    return target

def year_from_date(s: str) -> int:
    if not s:
        return 0
    try:
        return int(s.split("-")[0])
    except Exception:
        return 0

def novelty_score(row, known_artists: Set[str]) -> float:
    return 1.0 if row.get("primary_artist_id") not in known_artists else 0.0

def distance_score(row, centroid_vec: np.ndarray, scaler: StandardScaler) -> float:
    vec = np.array([row[c] for c in AUDIO_FEATURE_COLUMNS], dtype=float)
    vecs = scaler.transform([vec])[0]
    d = float(np.linalg.norm(vecs - centroid_vec))
    # invert distance (smaller is better)
    return 1.0 / (1.0 + d)

# --------------------------- MAIN WORKFLOW ----------------------------------

st.subheader("1) Input: Your Playlist(s)")
playlist_input = st.text_area(
    "Paste one or more Spotify playlist URLs/IDs (one per line)",
    placeholder="https://open.spotify.com/playlist/xxxxxxxx\nhttps://open.spotify.com/playlist/yyyyyyyy",
    height=100,
)

colA, colB, colC = st.columns(3)
with colA:
    auto_k = st.toggle("Auto-select number of moods (clusters)", value=True)
with colB:
    k_val = st.slider("If not auto, choose clusters", 2, 12, 5)
with colC:
    per_cluster = st.slider("Recommendations per mood", 10, 60, 25, step=5)

col1, col2, col3 = st.columns(3)
with col1:
    min_year, max_year = st.slider("Era filter (release year)", 1960, 2025, (1970, 2025))
with col2:
    target_obscurity = st.slider("Favor obscurity (lower popularity)", 0.0, 1.0, 0.4, help="Higher means prioritize less popular tracks.")
with col3:
    novelty_weight = st.slider("Favor new artists vs your playlist", 0.0, 1.0, 0.5)

lang_opts = st.multiselect(
    "Language emphasis (optional) — uses genre heuristics",
    options=["tamil", "malayalam", "hindi", "spanish", "english"],
    default=[],
)

run_btn = st.button("🔍 Generate Compatible Songs", type="primary", use_container_width=True)

if run_btn:
    if not sp:
        st.error("Please authenticate with Spotify first in the sidebar.")
        st.stop()
    if not playlist_input.strip():
        st.error("Please paste at least one playlist URL or ID.")
        st.stop()

    # Aggregate tracks from all playlists
    all_playlists = [ln.strip() for ln in playlist_input.strip().splitlines() if ln.strip()]
    with st.status("Fetching your playlist tracks…", expanded=False) as status:
        dfs = []
        for idx, pl in enumerate(all_playlists, start=1):
            st.write(f"Playlist {idx}: {pl}")
            dfp = get_playlist_tracks(sp, pl)
            dfs.append(dfp)
        df_tracks = pd.concat(dfs, ignore_index=True).drop_duplicates("track_id")
        status.update(label=f"Fetched {len(df_tracks)} unique tracks.")

    if df_tracks.empty:
        st.warning("No tracks found.")
        st.stop()

    # Audio features
    with st.status("Analyzing audio features…", expanded=False) as status:
        feats = get_audio_features(sp, df_tracks["track_id"].dropna().tolist())
        df = df_tracks.merge(feats, on="track_id", how="inner")
        # Drop rows missing critical features
        df = df.dropna(subset=[c for c in AUDIO_FEATURE_COLUMNS if c != "loudness"]).reset_index(drop=True)
        status.update(label=f"Analyzed {len(df)} tracks with features.")

    # Artist genres & language inference
    all_artist_ids = sorted({aid for row in df["artist_ids"].dropna() for aid in row})
    artist_genre_map = get_artist_genres(sp, all_artist_ids)
    df["primary_artist_id"] = df["artist_ids"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    df["artist_genres"] = df["primary_artist_id"].apply(lambda aid: artist_genre_map.get(aid, []))
    df["year"] = df["release_date"].apply(year_from_date)

    # Filter by year range
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].reset_index(drop=True)
    if df.empty:
        st.warning("After applying year filter, no tracks remain. Widen the range.")
        st.stop()

    # Clustering
    st.info("Clustering your playlist into emotional/mood groups…")
    if auto_k:
        k = choose_k(df[AUDIO_FEATURE_COLUMNS])
    else:
        k = k_val

    dfc, km, scaler = build_clusters(df, k)
    st.success(f"Identified {k} distinct mood clusters.")

    # Known baseline sets
    known_track_ids: Set[str] = set(df["track_id"].tolist())
    known_artist_ids: Set[str] = set(df["primary_artist_id"].dropna().tolist())

    # Language emphasis: build a set of allowed artists if languages selected
    allow_langs = set(lang_opts)

    def lang_ok(genres: List[str]) -> bool:
        if not allow_langs:
            return True
        gstr = " ".join(genres).lower()
        for lang in allow_langs:
            hints = LANG_GENRE_HINTS.get(lang, [])
            if any(h in gstr for h in hints):
                return True
        return False

    # Generate recommendations per cluster
    all_recs = []
    with st.status("Generating compatible songs per mood…", expanded=False) as status:
        for cl in sorted(dfc["cluster"].unique()):
            cdf = dfc[dfc["cluster"] == cl].copy()
            centroid_vec = km.cluster_centers_[cl]
            targets = centroid_targets(centroid_vec, scaler)

            # Seeds: a mix of tracks & artists from this cluster
            seed_tracks = cdf["track_id"].head(5).tolist()
            seed_artists = cdf["primary_artist_id"].dropna().head(5).tolist()
            if not seed_tracks and not seed_artists:
                continue

            # Multiple pulls to widen the pool
            pool = []
            tries = 4
            for t in range(tries):
                recs = sp.recommendations(
                    seed_tracks=seed_tracks[:5],
                    seed_artists=seed_artists[:5],
                    limit=min(100, per_cluster + 20),
                    target_acousticness=targets["acousticness"],
                    target_danceability=targets["danceability"],
                    target_energy=targets["energy"],
                    target_instrumentalness=targets["instrumentalness"],
                    target_liveness=targets["liveness"],
                    target_loudness=targets["loudness"],
                    target_speechiness=targets["speechiness"],
                    target_valence=targets["valence"],
                    target_tempo=targets["tempo"],
                )
                pool.extend(recs.get("tracks", []))
                time.sleep(0.2)

            # Normalize pool to DataFrame
            rows = []
            for tr in pool:
                if not tr or not tr.get("id"):
                    continue
                if tr["id"] in known_track_ids:
                    continue  # already in your playlist library subset
                artists = tr.get("artists", [])
                primary_artist = artists[0] if artists else {}
                aid = primary_artist.get("id")
                genres = artist_genre_map.get(aid, []) if aid else []
                if not lang_ok(genres):
                    continue
                album = tr.get("album", {})
                rows.append({
                    "rec_id": tr["id"],
                    "rec_name": tr["name"],
                    "rec_artist": ", ".join([a.get("name", "") for a in artists]),
                    "primary_artist_id": aid,
                    "artist_genres": genres,
                    "album": album.get("name"),
                    "year": year_from_date(album.get("release_date")),
                    "popularity": tr.get("popularity", 0),
                    "preview_url": tr.get("preview_url"),
                    # features for distance scoring
                    "acousticness": tr.get("acousticness"),
                    "danceability": tr.get("danceability"),
                    "energy": tr.get("energy"),
                    "instrumentalness": tr.get("instrumentalness"),
                    "liveness": tr.get("liveness"),
                    "loudness": tr.get("loudness"),
                    "speechiness": tr.get("speechiness"),
                    "valence": tr.get("valence"),
                    "tempo": tr.get("tempo"),
                    "cluster": cl,
                })

            rdf = pd.DataFrame.from_records(rows).drop_duplicates("rec_id")
            if rdf.empty:
                continue

            # Score combining distance to centroid, novelty, and obscurity
            rdf["novelty"] = rdf.apply(lambda r: novelty_score(r, known_artist_ids), axis=1)
            # fill features if None (Spotify recs usually include features; if missing, set neutral)
            for c in AUDIO_FEATURE_COLUMNS:
                if c not in rdf or rdf[c].isna().any():
                    rdf[c] = rdf[c].fillna(float(np.mean(cdf[c])))
            rdf["dist"] = rdf.apply(lambda r: distance_score(r, centroid_vec, scaler), axis=1)
            # Normalize components
            rdf["dist_norm"] = (rdf["dist"] - rdf["dist"].min()) / (rdf["dist"].max() - rdf["dist"].min() + 1e-9)
            rdf["obs_norm"] = 1.0 - (rdf["popularity"] / 100.0)
            alpha = float(novelty_weight)
            beta = float(target_obscurity)
            rdf["score"] = 0.60 * rdf["dist_norm"] + 0.40 * ((1-alpha) * (1 - rdf["novelty"]) + alpha * rdf["novelty"])  # balance fit & novelty
            rdf["score"] = (1 - beta) * rdf["score"] + beta * rdf["obs_norm"]

            # Era filter again (on recs)
            rdf = rdf[(rdf["year"] >= min_year) & (rdf["year"] <= max_year)]

            # Top N for this cluster
            rdf = rdf.sort_values("score", ascending=False).head(per_cluster)
            all_recs.append(rdf)

        status.update(label="Finished generating cluster recommendations.")

    if not all_recs:
        st.warning("No recommendations produced. Try lowering constraints or adding more playlists.")
        st.stop()

    recs_df = pd.concat(all_recs, ignore_index=True)

    # ------------------ DISPLAY ------------------
    st.subheader("2) Results: Your Compatible Songs")
    st.caption("Sorted within each mood cluster by match score, balancing fit, novelty, and tasteful obscurity.")

    # Per-cluster expandable tables
    for cl in sorted(recs_df["cluster"].unique()):
        with st.expander(f"Mood Cluster {cl}"):
            sub = recs_df[recs_df["cluster"] == cl].copy()
            sub_display = sub[["rec_name", "rec_artist", "album", "year", "popularity", "score", "preview_url", "artist_genres"]].rename(columns={
                "rec_name": "Track",
                "rec_artist": "Artist",
                "preview_url": "Preview",
            })
            st.dataframe(sub_display, use_container_width=True, hide_index=True)

    # Combined top list
    st.markdown("---")
    st.subheader("Top Picks Across All Moods")
    top_all = recs_df.sort_values(["score", "popularity"], ascending=[False, False]).head(100)
    st.dataframe(
        top_all[["rec_name", "rec_artist", "album", "year", "popularity", "score", "preview_url"]]
        .rename(columns={"rec_name": "Track", "rec_artist": "Artist", "preview_url": "Preview"}),
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV
    csv_name = f"music_soulmate_recs_{slugify(time.strftime('%Y%m%d_%H%M%S'))}.csv"
    st.download_button(
        label="💾 Download CSV",
        data=top_all.to_csv(index=False).encode("utf-8"),
        file_name=csv_name,
        mime="text/csv",
        use_container_width=True,
    )

    # ------------------ CREATE PLAYLIST ------------------
    st.markdown("---")
    st.subheader("3) Save to Spotify Playlist (Optional)")

    new_pl_name = st.text_input("New playlist name", value=f"Music Soulmate Picks — {time.strftime('%Y-%m-%d')}")
    new_pl_public = st.toggle("Public playlist?", value=False)
    picks_to_save = st.slider("How many tracks to add?", 10, 100, 50, step=10)
    save_btn = st.button("➕ Create Playlist & Add Tracks", type="primary")

    if save_btn:
        try:
            me = sp.me()
            user_id = me["id"]
            new_pl = sp.user_playlist_create(user=user_id, name=new_pl_name, public=new_pl_public, description="Curated by Music Soulmate Engine — taste-fit across moods")
            uri_batch = ["spotify:track:" + tid for tid in top_all.head(picks_to_save)["rec_id"].tolist()]
            # Spotify add in chunks of 100
            for i in range(0, len(uri_batch), 100):
                sp.playlist_add_items(new_pl["id"], uri_batch[i:i+100])
            st.success(f"Created playlist '{new_pl_name}' with {len(uri_batch)} tracks.")
        except Exception as e:
            st.error(f"Failed to create playlist: {e}")

# --------------------------- FOOTER ----------------------------------------
st.markdown("""
---
**Tips for better results**
- Paste multiple playlists (sad/happy/motivated, languages), the engine will learn all of them.
- Increase clusters for very large, eclectic libraries.
- Nudge *Favor obscurity* upward to surface hidden gems.
- Use *Language emphasis* to steer the engine toward Tamil/Malayalam/Hindi/Spanish/English scenes.
- Tighten or widen the *Era filter* to dig into specific decades.
""")
