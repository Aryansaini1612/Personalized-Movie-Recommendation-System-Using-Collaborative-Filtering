import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")
DATA_DIR = Path(__file__).resolve().parent / "data"
MOVIE_FILE = DATA_DIR / "movies.csv"
RATING_FILE = DATA_DIR / "ratings.csv"

DEMO_MOVIES = [
    (1, "The Shawshank Redemption", 1994, "Drama"), (2, "The Godfather", 1972, "Crime"),
    (3, "The Dark Knight", 2008, "Action"), (4, "Pulp Fiction", 1994, "Crime"),
    (5, "Forrest Gump", 1994, "Drama"), (6, "The Matrix", 1999, "Sci-Fi"),
    (7, "Goodfellas", 1990, "Crime"), (8, "Inception", 2010, "Sci-Fi"),
    (9, "The Lord of the Rings: The Fellowship of the Ring", 2001, "Fantasy"),
    (10, "Interstellar", 2014, "Sci-Fi"), (11, "The Silence of the Lambs", 1991, "Thriller"),
    (12, "The Green Mile", 1999, "Drama"), (13, "Parasite", 2019, "Thriller"),
    (14, "Spirited Away", 2001, "Animation"), (15, "Toy Story", 1995, "Animation"),
    (16, "The Grand Budapest Hotel", 2014, "Comedy"), (17, "La La Land", 2016, "Romance"),
    (18, "Titanic", 1997, "Romance"), (19, "Mad Max: Fury Road", 2015, "Action"),
    (20, "Spider-Man: Into the Spider-Verse", 2018, "Animation"), (21, "Whiplash", 2014, "Drama"),
    (22, "The Prestige", 2006, "Thriller"), (23, "Back to the Future", 1985, "Sci-Fi"),
    (24, "The Princess Bride", 1987, "Fantasy"), (25, "Knives Out", 2019, "Mystery"),
    (26, "The Social Network", 2010, "Drama"), (27, "Coco", 2017, "Animation"),
    (28, "The Avengers", 2012, "Action"), (29, "Get Out", 2017, "Horror"),
    (30, "Groundhog Day", 1993, "Comedy"), (31, "Amélie", 2001, "Romance"),
    (32, "The Lion King", 1994, "Animation"), (33, "Arrival", 2016, "Sci-Fi"),
    (34, "The Truman Show", 1998, "Comedy"), (35, "The Departed", 2006, "Crime"),
    (36, "Dune: Part Two", 2024, "Sci-Fi"), (37, "Everything Everywhere All at Once", 2022, "Fantasy"),
    (38, "Paddington 2", 2017, "Comedy"), (39, "The Conjuring", 2013, "Horror"),
    (40, "The Notebook", 2004, "Romance"),
]

@st.cache_data
def make_demo_ratings():
    rng = np.random.default_rng(24)
    tastes = [
        {"Drama", "Crime", "Thriller"}, {"Sci-Fi", "Action", "Fantasy"},
        {"Animation", "Fantasy", "Comedy"}, {"Romance", "Drama", "Comedy"},
        {"Crime", "Thriller", "Action"}, {"Sci-Fi", "Drama", "Mystery"},
        {"Animation", "Comedy", "Romance"}, {"Horror", "Thriller", "Mystery"},
        {"Action", "Sci-Fi", "Crime"}, {"Drama", "Romance", "Fantasy"},
        {"Comedy", "Animation", "Fantasy"}, {"Thriller", "Drama", "Crime"},
        {"Sci-Fi", "Fantasy", "Action"}, {"Romance", "Comedy", "Drama"},
        {"Horror", "Action", "Thriller"}, {"Crime", "Drama", "Mystery"},
        {"Animation", "Sci-Fi", "Comedy"}, {"Fantasy", "Romance", "Drama"},
    ]
    records = []
    for user_id, preferred in enumerate(tastes, start=1):
        for movie_id, _, _, genre in DEMO_MOVIES:
            if rng.random() < 0.58:
                rating = rng.normal(4.1 if genre in preferred else 3.0, 0.65)
                records.append((user_id, movie_id, float(np.clip(round(rating * 2) / 2, 1, 5))))
    return pd.DataFrame(records, columns=["user_id", "movie_id", "rating"])

def build_model(ratings):
    user_ids = ratings["user_id"].to_numpy(dtype=np.int32, copy=False)
    movie_ids = ratings["movie_id"].to_numpy(dtype=np.int32, copy=False)
    values = ratings["rating"].to_numpy(dtype=np.float32, copy=False)
    row_indices = user_ids - 1
    user_count = int(row_indices.max()) + 1
    movie_count = int(movie_ids.max()) + 1
    counts_by_user = np.bincount(row_indices, minlength=user_count)
    means_by_user = np.bincount(row_indices, weights=values, minlength=user_count) / counts_by_user
    deviations = values - means_by_user[row_indices].astype(np.float32)
    norms = np.sqrt(np.bincount(row_indices, weights=deviations * deviations, minlength=user_count))
    rating_counts = np.bincount(movie_ids, minlength=movie_count).astype(np.int32)
    matrix = csr_matrix((deviations, (row_indices, movie_ids)), shape=(user_count, movie_count), dtype=np.float32)
    return {"matrix": matrix, "user_norms": norms.astype(np.float32), "global_mean": float(values.mean()), "rating_counts": rating_counts, "rating_total": len(ratings)}

@st.cache_resource(show_spinner="Loading the movie ratings dataset...")
def load_project_data():
    if MOVIE_FILE.exists() and RATING_FILE.exists():
        movies = pd.read_csv(MOVIE_FILE, usecols=["movieId", "title", "genres"], dtype={"movieId": np.int32})
        ratings = pd.read_csv(RATING_FILE, usecols=["userId", "movieId", "rating"], dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
        movies = movies.rename(columns={"movieId": "movie_id"})
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).fillna("")
        movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        movies["genre"] = movies["genres"].fillna("(no genres listed)").str.replace("|", ", ", regex=False)
        movies = movies[["movie_id", "title", "year", "genre"]]
        ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
        model = build_model(ratings)
        del ratings
        return movies, model, "MovieLens 25M"
    movies = pd.DataFrame(DEMO_MOVIES, columns=["movie_id", "title", "year", "genre"])
    model = build_model(make_demo_ratings())
    return movies, model, "synthetic demo"

def recommend(profile, model, movies, neighbors=40):
    matrix = model["matrix"]
    profile_movie_ids = np.fromiter(profile.keys(), dtype=np.int32)
    profile_values = np.fromiter(profile.values(), dtype=np.float32)
    valid = (profile_movie_ids > 0) & (profile_movie_ids < matrix.shape[1])
    profile_movie_ids = profile_movie_ids[valid]
    profile_values = profile_values[valid]
    if len(profile_movie_ids) < 2:
        return pd.DataFrame()
    target = profile_values - model["global_mean"]
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-8:
        target = np.ones_like(profile_values)
        target_norm = float(np.linalg.norm(target))
    overlap = np.asarray(matrix[:, profile_movie_ids] @ target).reshape(-1)
    denominator = model["user_norms"] * target_norm
    similarities = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
    eligible = np.flatnonzero(similarities > 0)
    if not len(eligible):
        return pd.DataFrame()
    if len(eligible) > neighbors:
        chosen = eligible[np.argpartition(similarities[eligible], -neighbors)[-neighbors:]]
    else:
        chosen = eligible
    chosen = chosen[np.argsort(similarities[chosen])[::-1]]
    weights = similarities[chosen]
    weighted_deviations = np.zeros(matrix.shape[1], dtype=np.float32)
    total_weights = np.zeros(matrix.shape[1], dtype=np.float32)
    for user_index, weight in zip(chosen, weights):
        start, stop = matrix.indptr[user_index:user_index + 2]
        movie_ids = matrix.indices[start:stop]
        deviations = matrix.data[start:stop]
        np.add.at(weighted_deviations, movie_ids, deviations * weight)
        np.add.at(total_weights, movie_ids, weight)
    candidate_ids = np.flatnonzero(total_weights > 0)
    candidate_ids = candidate_ids[~np.isin(candidate_ids, profile_movie_ids)]
    if not len(candidate_ids):
        return pd.DataFrame()
    predictions = np.clip(np.mean(profile_values) + weighted_deviations[candidate_ids] / total_weights[candidate_ids], 0.5, 5.0)
    result = movies[movies["movie_id"].isin(candidate_ids)].copy()
    result["predicted_rating"] = result["movie_id"].map(dict(zip(candidate_ids, predictions)))
    result["rating_count"] = result["movie_id"].map(lambda movie_id: model["rating_counts"][movie_id] if movie_id < len(model["rating_counts"]) else 0)
    return result.sort_values(["predicted_rating", "rating_count"], ascending=False)

movies, model, data_source = load_project_data()
movie_lookup = movies.set_index("movie_id")
st.markdown("""
<style>
.stApp {background: #0b1020; color: #f3f4f6;}
.block-container {max-width: 1180px; padding-top: 2.2rem;}
[data-testid="stMetric"] {background: #151c31; border: 1px solid #252d45; padding: 16px; border-radius: 14px;}
.hero {padding: 26px 30px; border: 1px solid #2b3450; border-radius: 20px; background: linear-gradient(120deg,#171d34,#251b39); margin-bottom: 22px;}
.hero p {color: #bdc5da; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎬 CineMatch</h1><p>Find your next favorite film from viewers who rate movies like you.</p></div>', unsafe_allow_html=True)
st.caption(f"Using {data_source} data: {len(movies):,} movies and {model['rating_total']:,} ratings.")
left, right = st.columns([1.05, 1.55], gap="large")
if "profile" not in st.session_state:
    st.session_state.profile = {}
profile = st.session_state.profile
with left:
    st.subheader("Build your taste profile")
    query = st.text_input("Search movies", placeholder="Type a movie title")
    available_movies = movies[~movies["movie_id"].isin(profile)]
    if query.strip():
        matching_movies = available_movies[available_movies["title"].str.contains(re.escape(query.strip()), case=False, na=False)].head(40)
    else:
        popular_ids = np.argsort(model["rating_counts"])[::-1]
        matching_movies = available_movies[available_movies["movie_id"].isin(popular_ids[:40])]
    if not matching_movies.empty:
        options = matching_movies["movie_id"].tolist()
        def movie_label(movie_id):
            movie = movie_lookup.loc[movie_id]
            year = f" ({movie.year})" if movie.year else ""
            return f"{movie.title}{year} - {movie.genre}"
        chosen_movie = st.selectbox("Choose a movie to rate", options, format_func=movie_label)
        new_rating = st.slider("Your rating", 0.5, 5.0, 4.0, 0.5, key="new_movie_rating")
        if st.button("Add rating", use_container_width=True):
            profile[chosen_movie] = new_rating
            st.rerun()
    elif query.strip():
        st.info("No matching unrated movies. Try another title.")
    if profile:
        st.markdown("**Your ratings**")
        for movie_id in list(profile):
            movie = movie_lookup.loc[movie_id]
            label = f"{movie.title} ({movie.year})" if movie.year else movie.title
            rating_col, remove_col = st.columns([4, 1])
            profile[movie_id] = rating_col.slider(label, 0.5, 5.0, float(profile[movie_id]), 0.5, key=f"rating_{movie_id}")
            if remove_col.button("Remove", key=f"remove_{movie_id}"):
                del profile[movie_id]
                st.rerun()
    count_col, catalog_col = st.columns(2)
    count_col.metric("Your ratings", len(profile))
    catalog_col.metric("Movies", f"{len(movies):,}")
    with st.expander("How recommendations work"):
        st.write("We compare your ratings with mean-centered rating patterns from other users using cosine similarity. The closest positive neighbors vote on unseen movies, weighted by similarity.")
with right:
    st.subheader("Picked for you")
    if len(profile) < 2:
        st.info("Rate at least two movies to get personalized recommendations.")
    else:
        results = recommend(profile, model, movies)
        if results.empty:
            st.info("We need more overlap. Try rating a few more films from different genres.")
        else:
            st.caption("Predicted ratings from your closest user neighbors")
            for _, movie in results.head(8).iterrows():
                with st.container(border=True):
                    details, score = st.columns([4, 1])
                    details.markdown(f"**{movie.title}**")
                    year = f"{movie.year} · " if movie.year else ""
                    details.caption(f"{year}{movie.genre}")
                    score.metric("Match", f"{movie.predicted_rating:.1f} / 5")
st.divider()
st.caption("The bundled MovieLens data is for demonstration and research use. No real rating profile is saved.")
