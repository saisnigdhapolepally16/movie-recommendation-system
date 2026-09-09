"""
recommender.py
---------------
Builds the content-based recommendation model and exposes the reusable
recommend() function used by app.py (and usable directly from a Python
shell or notebook for testing).

Pipeline (continuing from preprocess.py's cleaned 'tags' column):

    tags (text) -> CountVectorizer -> sparse bag-of-words feature matrix
                -> cosine similarity via a Nearest-Neighbors index
                -> top-N most similar movies

Design note on memory efficiency
---------------------------------
With ~4,800 movies, a full pairwise cosine-similarity matrix would be
about 4800 x 4800 floats (~90+ MB), and that grows QUADRATICALLY with the
number of movies -- doubling the dataset would roughly quadruple that
matrix. Instead of ever building it, this module fits scikit-learn's
NearestNeighbors with metric="cosine" directly on the sparse feature
matrix. It computes cosine similarity only for the single movie being
queried, on demand, so memory usage stays proportional to the (sparse)
feature matrix itself rather than to the square of the number of movies.
"""

import difflib
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocess import PROCESSED_PATH, run_preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
FEATURE_MATRIX_PATH = os.path.join(MODELS_DIR, "feature_matrix.pkl")
NN_MODEL_PATH = os.path.join(MODELS_DIR, "nn_model.pkl")

# Cap the vocabulary size so the feature matrix stays small and training
# stays fast, while still capturing the most informative words/tokens.
MAX_FEATURES = 5000


# --------------------------------------------------------------------------
# Loading processed data
# --------------------------------------------------------------------------
def load_processed_movies(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned/feature-engineered DataFrame produced by
    preprocess.py, running preprocessing automatically the first time if
    it hasn't been generated yet.
    """
    if not os.path.exists(path):
        print("Processed dataset not found -- running preprocessing now ...")
        return run_preprocessing()
    return pd.read_pickle(path)


# --------------------------------------------------------------------------
# Building / persisting the model
# --------------------------------------------------------------------------
def build_model(df: pd.DataFrame):
    """Fit the CountVectorizer + NearestNeighbors (cosine) index on the
    movies' combined 'tags' text.

    Returns:
        vectorizer:     fitted CountVectorizer
        feature_matrix: sparse (n_movies x n_features) bag-of-words matrix
        nn_model:       fitted NearestNeighbors index (metric='cosine')
    """
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    feature_matrix = vectorizer.fit_transform(df["tags"])

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(feature_matrix)

    return vectorizer, feature_matrix, nn_model


def save_model(vectorizer, feature_matrix, nn_model):
    """Persist the trained vectorizer/feature matrix/index to models/ so
    the app doesn't need to refit them on every restart.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(FEATURE_MATRIX_PATH, "wb") as f:
        pickle.dump(feature_matrix, f)
    with open(NN_MODEL_PATH, "wb") as f:
        pickle.dump(nn_model, f)


def load_saved_model():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(FEATURE_MATRIX_PATH, "rb") as f:
        feature_matrix = pickle.load(f)
    with open(NN_MODEL_PATH, "rb") as f:
        nn_model = pickle.load(f)
    return vectorizer, feature_matrix, nn_model


def get_or_build_model(df: pd.DataFrame):
    """Load a previously trained vectorizer/index from disk if available
    and still consistent with the current dataset, otherwise (re)build it
    and persist the result for next time.
    """
    all_exist = all(os.path.exists(p) for p in (VECTORIZER_PATH, FEATURE_MATRIX_PATH, NN_MODEL_PATH))
    if all_exist:
        try:
            vectorizer, feature_matrix, nn_model = load_saved_model()
            if feature_matrix.shape[0] == len(df):
                return vectorizer, feature_matrix, nn_model
            print("Cached model doesn't match the current dataset size -- rebuilding ...")
        except Exception:
            print("Cached model files are unreadable -- rebuilding ...")

    vectorizer, feature_matrix, nn_model = build_model(df)
    save_model(vectorizer, feature_matrix, nn_model)
    return vectorizer, feature_matrix, nn_model


# --------------------------------------------------------------------------
# Movie lookup (with graceful "not found" handling)
# --------------------------------------------------------------------------
def find_movie_index(df: pd.DataFrame, movie_name: str):
    """Locate a movie by case-insensitive exact title match first, then
    fall back to a fuzzy match so small typos still resolve.

    Returns (index, matched_title), or (None, None) if nothing reasonably
    close was found.
    """
    titles = df["title"]
    lowered = titles.str.lower()
    query = movie_name.strip().lower()

    if not query:
        return None, None

    exact = df.index[lowered == query]
    if len(exact) > 0:
        return exact[0], titles.loc[exact[0]]

    close = difflib.get_close_matches(query, lowered.tolist(), n=1, cutoff=0.6)
    if close:
        match_idx = lowered[lowered == close[0]].index[0]
        return match_idx, titles.loc[match_idx]

    return None, None


def suggest_titles(df: pd.DataFrame, movie_name: str, n: int = 5):
    """Return up to n title suggestions close to an unmatched query, so the
    UI can offer 'did you mean ...?' instead of a bare error message.
    """
    query = movie_name.strip().lower()
    if not query:
        return []
    lowered = df["title"].str.lower().tolist()
    matches = difflib.get_close_matches(query, lowered, n=n, cutoff=0.4)
    seen = set()
    suggestions = []
    for m in matches:
        idx = lowered.index(m)
        title = df["title"].iloc[idx]
        if title not in seen:
            suggestions.append(title)
            seen.add(title)
    return suggestions


# --------------------------------------------------------------------------
# Explanation ("Why was this recommended?")
# --------------------------------------------------------------------------
def explain_recommendation(df: pd.DataFrame, base_idx: int, rec_idx: int):
    """Build a human-readable explanation of why `rec_idx` was recommended
    for `base_idx`, based on ACTUAL shared metadata -- never randomly
    generated. Returns a list of (category, [shared items]) tuples.
    """
    base = df.loc[base_idx]
    rec = df.loc[rec_idx]

    shared_genres = sorted(set(base["genres_list"]) & set(rec["genres_list"]))
    shared_director = sorted(set(base["director_list"]) & set(rec["director_list"]))
    shared_cast = sorted(set(base["cast_list"]) & set(rec["cast_list"]))
    shared_keywords = sorted(set(base["keywords_list"]) & set(rec["keywords_list"]))

    reasons = []
    if shared_genres:
        reasons.append(("Genre", shared_genres))
    if shared_director:
        reasons.append(("Director", shared_director))
    if shared_cast:
        reasons.append(("Cast", shared_cast))
    if shared_keywords:
        reasons.append(("Keywords", shared_keywords[:5]))  # keep the list short/readable
    return reasons


# --------------------------------------------------------------------------
# The main reusable recommendation function
# --------------------------------------------------------------------------
def recommend(
    movie_name: str,
    number_of_recommendations: int = 10,
    df: pd.DataFrame = None,
    vectorizer=None,
    feature_matrix=None,
    nn_model=None,
):
    """Return the top-N movies most similar to `movie_name`.

    Steps: find the selected movie -> retrieve its feature vector -> use
    cosine similarity (via the NearestNeighbors index) -> rank similar
    movies -> exclude the movie itself -> return the top N.

    Returns a dict:
        {
            "status": "ok" | "not_found",
            "matched_title": str or None,   # the title actually matched
            "results": pd.DataFrame,        # empty if not found
            "suggestions": list[str],       # only populated when not found
        }

    Pass in an already-loaded df/vectorizer/feature_matrix/nn_model (as
    app.py does via Streamlit caching) to avoid reloading/rebuilding them
    on every call. If omitted, they are loaded/built automatically --
    convenient for quick command-line testing, e.g.:

        python -c "from recommender import recommend; \\
            print(recommend('Avatar', 5)['results'][['title', 'similarity']])"
    """
    if df is None:
        df = load_processed_movies()
    if vectorizer is None or feature_matrix is None or nn_model is None:
        vectorizer, feature_matrix, nn_model = get_or_build_model(df)

    if number_of_recommendations < 1:
        number_of_recommendations = 1

    idx, matched_title = find_movie_index(df, movie_name)
    if idx is None:
        return {
            "status": "not_found",
            "matched_title": None,
            "results": df.iloc[0:0],
            "suggestions": suggest_titles(df, movie_name),
        }

    # Ask for a few extra neighbors since the query movie itself is always
    # returned as its own nearest neighbor (distance 0) and must be
    # excluded from the final results.
    k = min(number_of_recommendations + 1, feature_matrix.shape[0])
    distances, indices = nn_model.kneighbors(feature_matrix[idx], n_neighbors=k)
    distances, indices = distances.flatten(), indices.flatten()

    results = []
    for dist, rec_idx in zip(distances, indices):
        if rec_idx == idx:
            continue  # exclude the selected movie itself
        similarity = 1 - dist  # NearestNeighbors returns cosine DISTANCE = 1 - cosine similarity
        row = df.loc[rec_idx].to_dict()
        row["similarity"] = round(float(similarity), 4)
        row["reasons"] = explain_recommendation(df, idx, rec_idx)
        results.append(row)
        if len(results) == number_of_recommendations:
            break

    return {
        "status": "ok",
        "matched_title": matched_title,
        "results": pd.DataFrame(results),
        "suggestions": [],
    }


# --------------------------------------------------------------------------
# Illustrative Precision@K heuristic (see README "Model Evaluation" section)
# --------------------------------------------------------------------------
def precision_at_k_by_genre_overlap(df, movie_name, k, vectorizer, feature_matrix, nn_model):
    """A HEURISTIC, illustrative version of Precision@K for demo/teaching
    purposes only -- it is NOT a validated accuracy metric.

    Real Precision@K requires ground-truth relevance judgments (e.g. real
    users saying which recommendations they liked), which this project
    doesn't have. As a stand-in to demonstrate the *concept*, this treats
    a recommended movie as "relevant" if it shares at least one genre
    with the query movie, and reports what fraction of the top-K
    recommendations meet that bar. Do not present this number as real
    model accuracy -- see the README for a full explanation.
    """
    output = recommend(movie_name, k, df, vectorizer, feature_matrix, nn_model)
    if output["status"] != "ok" or output["results"].empty:
        return None

    idx, _ = find_movie_index(df, movie_name)
    query_genres = set(df.loc[idx, "genres_list"])
    if not query_genres:
        return None

    results = output["results"]
    relevant = results["genres_list"].apply(lambda g: len(query_genres & set(g)) > 0)
    return float(relevant.sum()) / float(len(results))


if __name__ == "__main__":
    # Quick manual smoke test when running this file directly:
    #   python recommender.py
    movies_df = load_processed_movies()
    vec, matrix, nn = get_or_build_model(movies_df)

    output = recommend("Avatar", 5, movies_df, vec, matrix, nn)
    print(f"Recommendations for: {output['matched_title']}")
    print(output["results"][["title", "similarity", "genres_display"]].to_string(index=False))
