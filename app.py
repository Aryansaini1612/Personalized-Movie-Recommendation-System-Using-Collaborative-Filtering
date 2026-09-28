import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

MOVIES = [
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
GENRES = sorted({movie[3] for movie in MOVIES})


@st.cache_data
def make_ratings():
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
        for movie_id, _, _, genre in MOVIES:
            if rng.random() < 0.58:
                rating = rng.normal(4.1 if genre in preferred else 3.0, 0.65)
                records.append((user_id, movie_id, float(np.clip(round(rating * 2) / 2, 1, 5))))
    return pd.DataFrame(records, columns=["user_id", "movie_id", "rating"])


def recommend(user_ratings, ratings, movies, neighbors=6):
    matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating")
    profile = pd.Series(user_ratings, dtype=float)
    common = profile.index.intersection(matrix.columns)
    if len(common) < 2:
        return pd.DataFrame()

    user_mean = profile.mean()
    centered_users = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)
    target = pd.Series(0.0, index=matrix.columns)
    target.loc[common] = profile.loc[common] - user_mean
    sims = cosine_similarity(target.to_numpy().reshape(1, -1), centered_users.to_numpy())[0]
    similarities = pd.Series(sims, index=matrix.index).clip(lower=0).sort_values(ascending=False)
    similarities = similarities[similarities > 0].head(neighbors)
    if similarities.empty:
        return pd.DataFrame()

    predictions = {}
    for movie_id in matrix.columns:
        if movie_id in profile.index:
            continue
        observed = matrix[movie_id].dropna().index.intersection(similarities.index)
        weights = similarities.loc[observed]
        if weights.sum() > 0:
            neighbor_means = matrix.loc[observed].mean(axis=1)
            deltas = matrix.loc[observed, movie_id] - neighbor_means
            predictions[movie_id] = float(np.clip(user_mean + np.dot(weights, deltas) / weights.sum(), 0.5, 5))
    result = movies[movies.movie_id.isin(predictions)].copy()
    result["predicted_rating"] = result.movie_id.map(predictions)
    return result.sort_values("predicted_rating", ascending=False)


movies = pd.DataFrame(MOVIES, columns=["movie_id", "title", "year", "genre"])
ratings = make_ratings()
st.markdown("""
<style>
.stApp {background: #0b1020; color: #f3f4f6;}
.block-container {max-width: 1180px; padding-top: 2.2rem;}
[data-testid="stMetric"] {background: #151c31; border: 1px solid #252d45; padding: 16px; border-radius: 14px;}
.hero {padding: 26px 30px; border: 1px solid #2b3450; border-radius: 20px; background: linear-gradient(120deg,#171d34,#251b39); margin-bottom: 22px;}
.hero p {color: #bdc5da; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎬 CineMatch</h1><p>Your next favorite film, found through people who rate movies like you.</p></div>', unsafe_allow_html=True)
left, right = st.columns([1.05, 1.55], gap="large")
with left:
    st.subheader("Build your taste profile")
    st.write("Rate a few films you’ve seen. More ratings help us find closer movie neighbors.")
    selected = st.multiselect("Choose movies", movies.title.tolist(), default=["The Matrix", "Inception", "Interstellar", "The Dark Knight"])
    profile = {}
    if selected:
        st.caption("Your rating (½ to 5 stars)")
        for title in selected:
            movie_id = int(movies.loc[movies.title == title, "movie_id"].iloc[0])
            profile[movie_id] = st.slider(title, 0.5, 5.0, 4.0, 0.5, key=f"rating_{movie_id}")
    count_col, genre_col = st.columns(2)
    count_col.metric("Demo raters", ratings.user_id.nunique())
    genre_col.metric("Genres", len(GENRES))
    with st.expander("How recommendations work"):
        st.write("We compare your mean-centered ratings with demo users using cosine similarity. The closest positive neighbors vote on movies you haven’t rated; their ratings are weighted by similarity.")

with right:
    st.subheader("Picked for you")
    if len(profile) < 2:
        st.info("Rate at least two movies to get personalized recommendations.")
    else:
        results = recommend(profile, ratings, movies)
        if results.empty:
            st.info("We need a little more overlap. Try rating two more films from different genres.")
        else:
            st.caption("Predicted ratings from your closest demo-user neighbors")
            for _, movie in results.head(8).iterrows():
                with st.container(border=True):
                    details, score = st.columns([4, 1])
                    details.markdown(f"**{movie.title}**  ")
                    details.caption(f"{int(movie.year)} · {movie.genre}")
                    score.metric("Match", f"{movie.predicted_rating:.1f} ★")

st.divider()
st.caption("Movie catalog and ratings are synthetic demo data. Replace them with a real ratings dataset for production use.")
