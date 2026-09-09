"""
app.py
------
Streamlit front-end for the Movie Recommendation System.

The user searches for or selects a movie, chooses how many
recommendations they want, and clicks "Recommend" to see similar movies
in a card grid -- each with its similarity score, genre, a short
overview, and an actual "why was this recommended?" explanation based on
shared metadata.

The recommendation algorithm itself (preprocess.py + recommender.py) runs
entirely locally with scikit-learn. The only optional external call is to
the TMDB API to fetch a poster image -- the app works perfectly well
without it.

Run with:
    streamlit run app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

from preprocess import DATA_DIR
from recommender import (
    get_or_build_model,
    load_processed_movies,
    precision_at_k_by_genre_overlap,
    recommend,
)

load_dotenv()  # reads a local .env file if present (see .env.example)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{id}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

st.set_page_config(page_title="Movie Recommendation System", page_icon="\U0001F3AC", layout="wide")


# --------------------------------------------------------------------------
# Cached loading: the model is built/loaded ONCE per running app process
# (st.cache_resource), not re-run on every widget interaction or page
# refresh. Poster lookups are cached separately (st.cache_data) since
# they're keyed by movie id and can safely expire after a while.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading movie data and building the recommendation model ...")
def load_data_and_model():
    df = load_processed_movies()
    vectorizer, feature_matrix, nn_model = get_or_build_model(df)
    return df, vectorizer, feature_matrix, nn_model


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster_url(movie_id: int, api_key: str):
    """Fetch a poster image URL from TMDB. Returns None on any failure
    (missing key, network error, movie not found, rate limit, ...) so the
    UI can fall back to a placeholder instead of breaking.
    """
    if not api_key:
        return None
    try:
        response = requests.get(
            TMDB_MOVIE_URL.format(id=movie_id),
            params={"api_key": api_key},
            timeout=5,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except Exception:
        return None


def render_poster(movie_id: int, title: str):
    """Show the TMDB poster if available, otherwise a lightweight
    placeholder -- the app must stay fully usable with no API key set."""
    poster_url = fetch_poster_url(movie_id, TMDB_API_KEY) if TMDB_API_KEY else None
    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="
                width:100%; aspect-ratio:2/3; border-radius:8px;
                background:linear-gradient(135deg,#2b2f3a,#454b5c);
                display:flex; align-items:center; justify-content:center;
                text-align:center; color:#d7dae0; font-size:0.85rem; padding:10px;
            ">
                <span>\U0001F3AC<br>{title}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommendation_card(row: dict):
    with st.container(border=True):
        render_poster(row["id"], row["title"])
        st.markdown(f"**{row['title']}**")
        st.caption(row.get("genres_display", "Unknown"))
        st.progress(min(max(row["similarity"], 0.0), 1.0), text=f"Similarity: {row['similarity'] * 100:.1f}%")

        overview = row.get("overview") or "No overview available."
        short_overview = overview if len(overview) <= 160 else overview[:157].rstrip() + "..."
        st.write(short_overview)

        reasons = row.get("reasons") or []
        with st.expander("Why was this recommended?"):
            if not reasons:
                st.write(
                    "No single shared genre/cast/director/keyword stood out, but the "
                    "movie's overall text profile (overview wording, themes, etc.) was "
                    "still the most similar among the dataset."
                )
            else:
                st.write("Recommended because of similarities in:")
                for category, items in reasons:
                    st.markdown(f"- **{category}:** {', '.join(items)}")


def render_recommend_tab(df, vectorizer, feature_matrix, nn_model):
    st.subheader("Find movies similar to one you like")

    mode = st.radio(
        "How would you like to choose a movie?",
        ["Pick from list", "Type a title"],
        horizontal=True,
        help="'Type a title' also demonstrates fuzzy matching / suggestions for misspelled titles.",
    )

    selected_movie = None
    if mode == "Pick from list":
        search_term = st.text_input("Search movies", placeholder="Type to filter, e.g. 'dark knight'")
        all_titles = df["title"].sort_values()
        if search_term.strip():
            filtered = all_titles[all_titles.str.contains(search_term, case=False, na=False)]
        else:
            filtered = all_titles
        filtered = filtered.head(300)  # keep the dropdown responsive

        if filtered.empty:
            st.info("No movies match your search.")
        else:
            selected_movie = st.selectbox("Select a movie", filtered.tolist())
    else:
        selected_movie = st.text_input("Type a movie title", placeholder="e.g. Avatar")

    num_recommendations = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

    if st.button("Recommend", type="primary"):
        if not selected_movie or not selected_movie.strip():
            st.warning("Please choose or type a movie title first.")
            return

        output = recommend(selected_movie, num_recommendations, df, vectorizer, feature_matrix, nn_model)

        if output["status"] == "not_found":
            st.error(f"Couldn't find a movie matching \"{selected_movie}\" in the dataset.")
            if output["suggestions"]:
                st.write("Did you mean:")
                for suggestion in output["suggestions"]:
                    st.markdown(f"- {suggestion}")
            else:
                st.write("Try a different spelling, or pick a title from the list instead.")
            return

        st.success(f"Showing movies similar to **{output['matched_title']}**")
        results = output["results"]

        cols_per_row = 5
        rows = [results.iloc[i : i + cols_per_row] for i in range(0, len(results), cols_per_row)]
        for row_chunk in rows:
            cols = st.columns(cols_per_row)
            for col, (_, movie_row) in zip(cols, row_chunk.iterrows()):
                with col:
                    render_recommendation_card(movie_row.to_dict())


def render_about_tab(df, vectorizer, feature_matrix, nn_model):
    st.subheader("How this system works")
    st.markdown(
        """
This is a **content-based** recommendation system: it recommends movies whose
*content* (overview, genres, keywords, cast, director) is textually similar to
the movie you picked. It does **not** use other users' ratings or behavior
(that would be *collaborative filtering* -- see Limitations below).

**Pipeline:** Raw TMDB CSVs → clean & merge → parse genres/keywords/cast/crew →
build a combined "tags" text per movie → `CountVectorizer` turns that text into
numeric vectors → cosine similarity ranks the closest vectors → top-N results.
        """
    )

    st.subheader("Evaluating a recommender system")
    st.markdown(
        """
Recommendation systems don't have a single "accuracy" number the way a digit
classifier does -- there's no ground-truth label for "the correct
recommendation." Instead, they're evaluated with different tools:

- **Similarity analysis** — inspect the actual cosine similarity scores
  returned for a query (shown as the progress bar on each card above). Scores
  close to 1.0 mean the two movies' tag vectors are nearly identical; scores
  near 0 mean almost no shared vocabulary.
- **Example recommendation inspection** — manually check a handful of
  well-known movies (e.g. a sequel, a franchise entry, a director's other
  films) and confirm the recommendations make intuitive sense. This is the
  most common way small content-based systems are sanity-checked in practice.
- **Precision@K** — of the top K recommendations shown to a user, what
  fraction were actually "relevant"? In a production system, "relevant" is
  determined by real user feedback (clicks, ratings, watches) which this
  project doesn't have. The illustrative demo below approximates "relevant"
  as *shares at least one genre with the query movie* purely to demonstrate
  the concept — **this is a heuristic for teaching purposes, not a real
  accuracy metric, and should never be presented as one.**
        """
    )

    st.markdown("**Illustrative Precision@K demo** (heuristic: genre overlap = \"relevant\")")
    demo_movie = st.selectbox(
        "Pick a movie to illustrate Precision@K for",
        df["title"].sort_values().tolist(),
        index=0,
        key="precision_demo_movie",
    )
    demo_k = st.slider("K", min_value=3, max_value=20, value=10, key="precision_demo_k")
    score = precision_at_k_by_genre_overlap(df, demo_movie, demo_k, vectorizer, feature_matrix, nn_model)
    if score is None:
        st.info("Not enough genre data for this movie to compute the heuristic.")
    else:
        st.metric(f"Heuristic Precision@{demo_k}", f"{score * 100:.0f}%")
        st.caption(
            "Fraction of the top-K recommendations that share at least one genre with "
            f"\"{demo_movie}\". Again: a teaching illustration, not a validated accuracy score."
        )

    st.subheader("Limitations of content-based filtering")
    st.markdown(
        """
- **No collaborative signal** — it never learns from what other users liked,
  so it can miss non-obvious but popular pairings that collaborative
  filtering would catch (e.g. two movies with different genres/cast that
  fans of one also tend to enjoy).
- **Cold start for sparse metadata** — a movie with a short/missing overview,
  no keywords, and an unlisted cast produces a weak "tags" vector and
  therefore weak recommendations.
- **Overspecialization** — because it only looks at content similarity, it
  tends to recommend more of the same (sequels, same genre/director) rather
  than surfacing serendipitous, different-but-still-enjoyable movies.
- **No notion of quality** — a highly similar but poorly-reviewed movie can
  rank above a slightly-less-similar but excellent one, since similarity and
  rating are independent of each other here.
- **Static vocabulary** — `CountVectorizer` only knows the words present in
  this dataset's overviews/keywords at the time the model was built; a
  synonym or a newly coined term it never saw won't be matched.
        """
    )


def render_missing_dataset_instructions():
    st.error("The TMDB dataset was not found.")
    st.markdown(
        f"""
        This app needs `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside the
        `data/` folder before it can build the recommendation model.

        **To fix this:**
        1. Download the TMDB 5000 Movie Dataset (see the README for links).
        2. Place both CSV files in:
           ```
           {DATA_DIR}
           ```
        3. Restart the app:
           ```bash
           streamlit run app.py
           ```
        """
    )


def main():
    st.title("Movie Recommendation System")
    st.caption(
        "Content-based recommendations powered by TF/CountVectorizer + cosine similarity "
        "over movie overviews, genres, keywords, cast, and director — all computed locally."
    )

    try:
        df, vectorizer, feature_matrix, nn_model = load_data_and_model()
    except FileNotFoundError:
        render_missing_dataset_instructions()
        st.stop()
        return

    if not TMDB_API_KEY:
        st.info(
            "No TMDB_API_KEY found — posters will show as placeholders. "
            "The recommendation engine itself is unaffected. See .env.example to enable posters.",
            icon="\U0001F5BC️",
        )

    tab_recommend, tab_about = st.tabs(["Recommend", "About & Evaluation"])
    with tab_recommend:
        render_recommend_tab(df, vectorizer, feature_matrix, nn_model)
    with tab_about:
        render_about_tab(df, vectorizer, feature_matrix, nn_model)


if __name__ == "__main__":
    main()
