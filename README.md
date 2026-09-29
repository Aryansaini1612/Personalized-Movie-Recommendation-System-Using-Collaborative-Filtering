# CineMatch: Personalized Movie Recommendation System

An interactive movie recommender using user-based collaborative filtering and the MovieLens 25M dataset.

## Run it

Requires Python 3.9 or later.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The app looks for `data/movies.csv` and `data/ratings.csv` beside `app.py`. Extract these files from the supplied archive into the `data/` folder. If either file is missing, the app starts with synthetic demo data. The first run with the full dataset may take a few minutes while the sparse ratings model is built and cached.

## Project files

- `app.py` - Streamlit interface, MovieLens loader, and sparse collaborative filtering model.
- `requirements.txt` - Python dependencies.
- `data/` - Local MovieLens CSV files; excluded from Git because `ratings.csv` is about 678 MB.

## Recommendation method

1. Mean-center each user's ratings.
2. Compare the visitor's rated movies with other users using cosine similarity.
3. Predict unrated movie scores from the 40 closest positive neighbors, weighted by similarity.

The app requires at least two rated movies and returns up to eight recommendations. It does not save personal rating profiles. The dataset stays local and is not uploaded to this repository.
