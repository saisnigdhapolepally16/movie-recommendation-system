"""
preprocess.py
-------------
Data cleaning and feature engineering for the Movie Recommendation System.

Pipeline:
    Raw TMDB CSVs -> merge -> handle missing values/duplicates -> parse
    JSON-like columns (genres, keywords, cast, crew) -> extract top cast
    + director -> normalize multi-word tokens -> build a combined 'tags'
    feature per movie -> save the cleaned DataFrame to disk.

This module only prepares clean, feature-engineered DATA. Vectorization
(CountVectorizer) and similarity search live in recommender.py, so each
file has a single clear responsibility: a student can explain preprocess.py
without needing to talk about the ML model at all, and vice versa.

Run standalone with:
    python preprocess.py
"""

import ast
import os

import pandas as pd

# --------------------------------------------------------------------------
# Paths & configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
PROCESSED_PATH = os.path.join(MODELS_DIR, "processed_movies.pkl")

# How many top-billed cast members to keep per movie. The TMDB 'cast' JSON
# lists actors roughly in order of prominence, so the first few are the
# lead/most important actors.
TOP_CAST_COUNT = 3


# --------------------------------------------------------------------------
# Loading & merging
# --------------------------------------------------------------------------
def load_raw_data(movies_path=MOVIES_CSV, credits_path=CREDITS_CSV):
    """Load the two raw TMDB CSVs, with a clear error if they're missing."""
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(
            "TMDB dataset not found.\n"
            f"Expected files at:\n  {movies_path}\n  {credits_path}\n"
            "Please download 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' "
            "and place them in the data/ folder. See the README for download links."
        )
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits


def merge_datasets(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Join the movies and credits tables into one row per movie.

    Both files describe the same movie via different id columns
    (movies.id == credits.movie_id). We merge on that numeric id rather
    than on 'title', since titles can repeat or differ slightly between
    the two files, while the id is a stable TMDB identifier.
    """
    credits_renamed = credits.rename(columns={"title": "credits_title"})
    merged = movies.merge(credits_renamed, left_on="id", right_on="movie_id")
    merged = merged.drop(columns=["movie_id", "credits_title"])
    return merged


# --------------------------------------------------------------------------
# Parsing the JSON-like columns
# --------------------------------------------------------------------------
def parse_names(json_like_str, key="name", limit=None):
    """Parse a JSON-like string such as
        '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    (the format TMDB stores 'genres', 'keywords', and 'cast' in) into a
    plain Python list of names, e.g. ["Action", "Adventure"].

    Uses ast.literal_eval (NOT eval) since it only parses Python literals
    and can't execute arbitrary code -- important when parsing data from
    a CSV file. Returns [] for missing/malformed values instead of raising,
    so a single bad row can't crash the whole pipeline.
    """
    if not isinstance(json_like_str, str):
        return []
    try:
        items = ast.literal_eval(json_like_str)
    except (ValueError, SyntaxError):
        return []
    names = [str(item.get(key, "")) for item in items if isinstance(item, dict)]
    names = [n for n in names if n]
    if limit is not None:
        names = names[:limit]
    return names


def extract_director(crew_json_like_str):
    """Pull the director's name out of the 'crew' column. The crew list
    contains many roles (producer, editor, composer, ...); we specifically
    look for the entry whose 'job' is 'Director'. Returns [] if none is
    listed, so it can be treated uniformly with the other list-valued
    features (genres, keywords, cast).
    """
    if not isinstance(crew_json_like_str, str):
        return []
    try:
        crew = ast.literal_eval(crew_json_like_str)
    except (ValueError, SyntaxError):
        return []
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = member.get("name")
            return [name] if name else []
    return []


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def normalize_token(token: str) -> str:
    """Collapse a multi-word entity into a single token by removing spaces
    and lowercasing, e.g.:
        "Science Fiction"   -> "sciencefiction"
        "Sam Worthington"   -> "samworthington"
        "Christopher Nolan" -> "christophernolan"

    Without this step, a CountVectorizer would split "Science Fiction"
    into the two independent words "science" and "fiction". That would
    incorrectly link this movie to any other movie whose overview merely
    contains the common word "science", or conflate two different actors
    who happen to share a first name. Collapsing each named entity into
    one token keeps these relationships meaningful.
    """
    return token.replace(" ", "").lower()


def build_tags(row) -> str:
    """Combine all selected metadata into one space-separated string per
    movie -- the 'bag of words' document that CountVectorizer will turn
    into a numeric feature vector.
    """
    overview_words = str(row["overview"]).lower().split()
    genre_tokens = [normalize_token(g) for g in row["genres_list"]]
    keyword_tokens = [normalize_token(k) for k in row["keywords_list"]]
    cast_tokens = [normalize_token(c) for c in row["cast_list"]]
    director_tokens = [normalize_token(d) for d in row["director_list"]]

    all_tokens = overview_words + genre_tokens + keyword_tokens + cast_tokens + director_tokens
    return " ".join(all_tokens)


def clean_and_engineer_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning + feature engineering pipeline on the merged
    movies+credits table and return a compact DataFrame ready for
    vectorization.
    """
    df = merged.copy()

    # --- Handle missing values -------------------------------------------
    # A missing overview/tagline just becomes an empty string rather than
    # dropping the movie outright, since genres/cast/director alone can
    # still produce a useful recommendation signal.
    df["overview"] = df["overview"].fillna("")
    if "tagline" in df.columns:
        df["tagline"] = df["tagline"].fillna("")
    else:
        df["tagline"] = ""
    # A movie with no title or id, however, can't be shown or matched, so
    # those rows are dropped.
    df = df.dropna(subset=["title", "id"])

    # --- Remove duplicate movies ------------------------------------------
    # The same movie can occasionally appear more than once (re-releases,
    # data entry duplicates). Keep the first occurrence of each id and
    # each title.
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # --- Parse JSON-like metadata columns ----------------------------------
    df["genres_list"] = df["genres"].apply(lambda x: parse_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: parse_names(x))
    df["cast_list"] = df["cast"].apply(lambda x: parse_names(x, limit=TOP_CAST_COUNT))
    df["director_list"] = df["crew"].apply(extract_director)

    # Human-readable versions (kept WITH spaces) purely for display in the
    # Streamlit UI -- the normalized, space-free versions are only used
    # internally for vectorization.
    df["genres_display"] = df["genres_list"].apply(lambda lst: ", ".join(lst) if lst else "Unknown")
    df["cast_display"] = df["cast_list"].apply(lambda lst: ", ".join(lst) if lst else "Unknown")
    df["director_display"] = df["director_list"].apply(lambda lst: ", ".join(lst) if lst else "Unknown")

    # --- Build the combined text feature used for vectorization ------------
    df["tags"] = df.apply(build_tags, axis=1)

    # Drop movies that ended up with an essentially empty tag string (no
    # overview, genres, keywords, cast, or director at all) -- there is no
    # usable signal to ever recommend them from or to.
    df = df[df["tags"].str.strip().str.len() > 0].reset_index(drop=True)

    keep_columns = [
        "id", "title", "overview", "tagline",
        "genres_list", "keywords_list", "cast_list", "director_list",
        "genres_display", "cast_display", "director_display",
        "release_date", "vote_average", "vote_count", "popularity",
        "tags",
    ]
    return df[keep_columns]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_preprocessing(movies_path=MOVIES_CSV, credits_path=CREDITS_CSV, save_path=PROCESSED_PATH):
    """Run the full pipeline end-to-end and persist the cleaned DataFrame."""
    movies, credits = load_raw_data(movies_path, credits_path)
    merged = merge_datasets(movies, credits)
    processed = clean_and_engineer_features(merged)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed.to_pickle(save_path)

    print(f"Loaded {len(movies)} movies and {len(credits)} credit records.")
    print(f"After cleaning/deduplication: {len(processed)} movies remain.")
    print(f"Saved processed dataset -> {save_path}")
    return processed


if __name__ == "__main__":
    run_preprocessing()
