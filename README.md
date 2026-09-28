# CineMatch: Personalized Movie Recommendation System

An interactive movie recommender demonstrating **user-based collaborative filtering**. Rate a few films and the app finds similar demo users, then estimates your ratings for unseen movies.

## Run locally

Requires Python 3.9 or later.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## How it works

1. Build a user–movie ratings matrix.
2. Center ratings by each user's average.
3. Compare your profile with demo users using cosine similarity.
4. Predict unseen movie ratings from the top positive neighbors, weighted by similarity.

The project uses deterministic synthetic demo ratings so it runs without a dataset download or API key. Replace these ratings with a real dataset such as MovieLens for production use. Evaluate with a held-out split using RMSE or MAE; demo scores are not real audience ratings.
