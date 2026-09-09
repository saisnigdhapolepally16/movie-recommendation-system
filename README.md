# Movie Recommendation System

A complete, runnable content-based movie recommendation system built with Python, pandas, scikit-learn, and Streamlit — using the TMDB 5000 Movie dataset. Pick a movie, and it recommends similar ones, with a real explanation of *why*, computed entirely from local data (no external AI API is used for the recommendations themselves).

## 1. Project Objective

Build a web application where a user selects or searches for a movie and receives a ranked list of similar movies, along with the similarity score and the actual shared metadata (genre, director, cast, keywords) that drove each recommendation. The goal is to demonstrate, end-to-end and transparently, how a content-based recommender actually works — not to call an external AI service and treat it as a black box.

## 2. Dataset

This project uses the **TMDB 5000 Movie Dataset**, a well-known public dataset originally sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/) and widely mirrored for ML coursework. It consists of two CSV files:

- `tmdb_5000_movies.csv` — budget, genres, keywords, overview, popularity, release date, revenue, runtime, title, vote average/count, and more (4,803 movies).
- `tmdb_5000_credits.csv` — cast and crew (director, producer, etc.) for each movie, keyed by `movie_id`.

Both files are already included in `data/` in this project. If you need to re-download them, search for "TMDB 5000 Movie Dataset" — it's available on Kaggle and mirrored in several public GitHub repositories.

**Note:** TMDB's `genres`, `keywords`, `cast`, and `crew` columns are stored as *JSON-like strings* inside the CSV, e.g. `"[{"id": 28, "name": "Action"}, ...]"`. `preprocess.py` parses these safely with `ast.literal_eval` (not `eval`, which would execute arbitrary code).

## 3. Recommendation System Concept

There are two broad families of recommender systems:

- **Content-based filtering** (used here): recommends items similar *in content* to something the user already likes — similar genres, cast, plot, etc. Works from day one, even for a brand-new user, since it needs no history — only the item's own metadata.
- **Collaborative filtering** (not used here): recommends items that *other, similar users* also liked, based on rating/interaction history across many users. It can surface non-obvious connections but needs a large base of user behavior data to work well, and can't recommend brand-new items no one has interacted with yet (the "cold start" problem).

This project implements **content-based filtering**, since it's self-contained, explainable, and doesn't require any user history or ratings database.

## 4. Feature Engineering

For each movie, `preprocess.py` builds one combined text "document" (called `tags`) from:

- **Overview** — the plot summary, lowercased and split into words.
- **Genres** — e.g. `["Action", "Adventure", "Science Fiction"]`.
- **Keywords** — TMDB's own tags for themes/subjects, e.g. `["space", "alien", "culture clash"]`.
- **Top cast** — the 3 most prominent actors (TMDB lists cast roughly in order of billing/importance).
- **Director** — extracted from the `crew` column by finding the entry whose `job` is `"Director"`.

**Multi-word normalization:** before combining, every genre/keyword/cast/director entry has its spaces removed and is lowercased — e.g. `"Science Fiction"` → `"sciencefiction"`, `"Sam Worthington"` → `"samworthington"`, `"Christopher Nolan"` → `"christophernolan"`. Without this, a vectorizer would split `"Science Fiction"` into the two independent words *science* and *fiction*, incorrectly linking this movie to any unrelated movie whose overview merely mentions "science," or conflating two different people who share a first name. Collapsing each named entity into a single token keeps every relationship meaningful.

Data cleaning also handles:

- **Missing values** — a missing overview/tagline becomes `""` rather than dropping the movie (genre/cast/director alone can still drive a decent recommendation); rows missing a title or id are dropped since they can't be shown or matched.
- **Duplicates** — movies are de-duplicated by both `id` and `title`.
- **Empty-signal movies** — a movie left with zero words in its final `tags` (no overview, genres, keywords, cast, or director at all) is dropped, since there's no content to compare it against.

## 5. Vectorization

The combined `tags` text for all ~4,800 movies is converted into numeric vectors with scikit-learn's **`CountVectorizer`** — a classic "bag of words" model:

- Every distinct word/token across all movies becomes one column (capped at the 5,000 most frequent, via `max_features=5000`).
- English stop words ("the", "and", "is", ...) are removed automatically (`stop_words="english"`).
- Each movie becomes a row vector where each value is *how many times* that word appears in its `tags`.

The result is a sparse matrix of shape `(4800 movies, ≤5000 words)` — sparse because the overwhelming majority of the 5,000 words don't appear in any given movie's tags, so scikit-learn only stores the non-zero entries.

## 6. Cosine Similarity

Cosine similarity measures how similar two vectors are by the *angle* between them, ignoring magnitude:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

- `A · B` is the dot product (sum of element-wise products).
- `||A||` and `||B||` are the vectors' magnitudes (Euclidean length).
- The result ranges from **0** (completely different/orthogonal — no shared words) to **1** (pointing in exactly the same direction — identical word-usage pattern).

This matters more than raw word-overlap counts because it's **length-independent**: a movie with a long overview naturally has more total words than one with a short overview, but cosine similarity only cares about the *proportional* word pattern, not the absolute counts. A short and a long overview that both emphasize the same words/genres/cast will still score highly similar.

### A tiny worked example

Suppose after vectorizing on a tiny 4-word vocabulary `[action, space, alien, drama]`, two movies have these counts:

```
Movie A: [2, 1, 1, 0]
Movie B: [1, 1, 1, 0]
```

```
A · B      = (2*1) + (1*1) + (1*1) + (0*0) = 4
||A||      = sqrt(2² + 1² + 1² + 0²) = sqrt(6) ≈ 2.449
||B||      = sqrt(1² + 1² + 1² + 0²) = sqrt(3) ≈ 1.732

cosine_similarity(A, B) = 4 / (2.449 * 1.732) ≈ 4 / 4.243 ≈ 0.943
```

A similarity of **~0.94** means these two movies' word-usage patterns are very close (both are action/space/alien-heavy, neither mentions drama).

### Why NearestNeighbors instead of a full similarity matrix

A naive implementation would precompute a full `4800 x 4800` matrix of every pairwise similarity. That's about 23 million numbers — manageable at this dataset size, but it grows **quadratically**: double the movies, and the matrix grows roughly 4x. Instead, `recommender.py` fits scikit-learn's **`NearestNeighbors(metric="cosine")`** directly on the sparse feature matrix. It computes cosine similarity only for the one movie you actually query, on demand, so memory use scales with the (sparse) feature matrix itself rather than the square of the number of movies — the same underlying math, computed lazily and efficiently.

## 7. Application Architecture

```
Movie Dataset (data/*.csv)
        │
        ▼
  preprocess.py     — clean, merge, parse JSON columns, engineer 'tags'
        │
        ▼
 models/processed_movies.pkl   (cleaned DataFrame)
        │
        ▼
  recommender.py    — CountVectorizer → NearestNeighbors(cosine) → recommend()
        │
        ▼
 models/{vectorizer, feature_matrix, nn_model}.pkl   (cached, reused on restart)
        │
        ▼
     app.py         — Streamlit UI: search, select, recommend, display, explain
```

`app.py` never touches the ML internals directly — it calls `recommend()` from `recommender.py`, which is deliberately a single, self-contained, reusable function (usable from a script, notebook, or the app alike).

## 8. Project Structure

```
movie-recommendation-system/
├── app.py                    # Streamlit web application (UI + orchestration)
├── recommender.py            # Vectorization, similarity index, recommend()
├── preprocess.py             # Data cleaning & feature engineering
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── models/
│   ├── processed_movies.pkl  # cleaned DataFrame (from preprocess.py)
│   ├── vectorizer.pkl        # fitted CountVectorizer
│   ├── feature_matrix.pkl    # sparse bag-of-words matrix
│   └── nn_model.pkl          # fitted NearestNeighbors (cosine) index
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 9. Installation

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd movie-recommendation-system
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 10. Dataset Placement

The dataset is already included under `data/`. If you're setting the project up fresh, make sure these two files exist before running anything:

```
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

## 11. Running the Project

Preprocessing and model-building both run automatically the first time you launch the app — but you can also run them explicitly:

```bash
python preprocess.py     # cleans the data -> models/processed_movies.pkl
python recommender.py    # builds the vectorizer/similarity index, runs a quick smoke test
streamlit run app.py     # launches the web app
```

The app opens at `http://localhost:8501`. If the dataset CSVs are missing, it shows clear on-screen setup instructions instead of crashing.

## 12. Optional TMDB Poster Setup

Movie posters are entirely optional — the app is fully functional without them (a placeholder card is shown instead). To enable real posters:

1. Create a free TMDB account and generate an API key: <https://www.themoviedb.org/settings/api>
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and set your key:
   ```
   TMDB_API_KEY=your_real_key_here
   ```
4. Restart the app. `.env` is gitignored, so your key is never committed. **No key is ever hard-coded in the source.**

## 13. Example Recommendations

Running `python recommender.py` (or the app) for a few well-known movies:

**Avatar** → Titan A.E., Small Soldiers, Ender's Game, Independence Day, Aliens vs Predator: Requiem — all sci-fi/action/adventure titles, matching Avatar's genre and thematic keywords.

**The Dark Knight Rises** → The Dark Knight, Batman Begins, Batman Returns, Batman, Batman Forever — every one of these shares either the director (Christopher Nolan, for the first two), the lead actor (Christian Bale), or Batman/Gotham-related keywords, all surfaced explicitly in the "Why was this recommended?" panel.

These results demonstrate the system correctly picking up on genre, director, cast, and keyword signals — not just superficial title similarity.

## 14. Model Evaluation

Recommender systems don't have a single "accuracy" number the way a classifier does — there's no ground-truth "correct" recommendation to check against. This project's "About & Evaluation" tab covers the standard ways such systems are actually assessed:

- **Similarity analysis** — inspecting the raw cosine similarity scores returned for a query (shown on every recommendation card).
- **Example recommendation inspection** — manually spot-checking well-known movies (franchises, a director's filmography) and confirming the results make intuitive sense — the most common way small content-based systems are sanity-checked in practice.
- **Precision@K (concept)** — of the top K recommendations, what fraction are actually "relevant"? In production this requires real user feedback (clicks, ratings). This project includes an illustrative, clearly-labeled heuristic (genre overlap as a proxy for "relevant") purely to demonstrate the *concept* — it is **not** presented as a real accuracy metric anywhere in the app.
- **Limitations of content-based filtering** — no collaborative signal, cold start for sparsely-described movies, tends toward overspecialization (more of the same rather than serendipity), no notion of movie quality/rating, and a vocabulary frozen at build time. Full discussion is in the app's "About & Evaluation" tab.

This project intentionally does **not** display any fabricated accuracy percentage.

## 15. Limitations

- Recommendations are driven purely by text/metadata similarity, not by user taste, ratings, or box-office success.
- A movie with sparse metadata (short overview, no keywords) will generate weak recommendations.
- The vocabulary is fixed at the time `CountVectorizer` is fit; a synonym or newly coined term never seen in this dataset won't match.
- No personalization: two different users searching the same movie get identical results.

## 16. Future Improvements

- **Collaborative filtering** — incorporate a user-item ratings matrix (e.g. via matrix factorization / SVD) to recommend based on what similar *users* liked, not just similar *content*.
- **Hybrid recommendation** — blend content-based and collaborative scores (e.g. weighted average) to get the best of both: cold-start coverage from content-based, and taste-pattern discovery from collaborative.
- **User-personalized recommendations** — track a user's viewing/rating history and bias recommendations toward their historical preferences (genre weighting, favorite directors/actors).
- **TF-IDF instead of raw counts** — down-weight very common words automatically instead of relying solely on the stop-word list.
- **Feature weighting** — currently overview words, genres, keywords, cast, and director are all weighted equally by simple concatenation; tuning relative importance (e.g. repeating the director token to boost its weight) could improve results.
- **Stemming/lemmatization** on overview text (e.g. "running"/"ran"/"runs" → "run") to reduce vocabulary fragmentation.
- **Public deployment** (Streamlit Community Cloud, Docker) instead of local-only.
