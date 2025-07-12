import pandas as pd
from pathlib import Path
import textstat
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
import joblib

# Load the ground truth
df = pd.read_csv("./data/train.csv")
print("Loaded train.csv with shape:", df.shape)

# Pad the ID to match folder names (e.g., 0 -> article_0000)
def id_to_folder_name(i):
    return f"article_{i:04d}"

def load_texts(row):
    folder = Path("./data/train") / id_to_folder_name(row["id"])
    with open(folder / "file_1.txt", "r", encoding="utf-8") as f1:
        text1 = f1.read()
    with open(folder / "file_2.txt", "r", encoding="utf-8") as f2:
        text2 = f2.read()
    return pd.Series([text1, text2])

# === Feature Engineering ===
def compute_tfidf_cosine(texts1, texts2):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    texts1 = list(texts1)
    texts2 = list(texts2)
    combined = texts1 + texts2
    tfidf = vectorizer.fit_transform(combined)
    tfidf_1 = tfidf[:len(texts1)]
    tfidf_2 = tfidf[len(texts1):]
    return np.array([cosine_similarity(tfidf_1[i], tfidf_2[i])[0][0] for i in range(len(texts1))])

def extract_features(df, emb_1, emb_2):
    df["len_1"] = df["text_1"].str.len()
    df["len_2"] = df["text_2"].str.len()
    df["wordcount_1"] = df["text_1"].str.split().apply(len)
    df["wordcount_2"] = df["text_2"].str.split().apply(len)
    df["len_diff"] = df["len_1"] - df["len_2"]
    df["wordcount_diff"] = df["wordcount_1"] - df["wordcount_2"]
    df["readability_1"] = df["text_1"].apply(textstat.flesch_reading_ease)
    df["readability_2"] = df["text_2"].apply(textstat.flesch_reading_ease)
    df["avg_sentence_length_1"] = df["text_1"].apply(lambda x: np.mean([len(s.split()) for s in x.split('.') if s]))
    df["avg_sentence_length_2"] = df["text_2"].apply(lambda x: np.mean([len(s.split()) for s in x.split('.') if s]))
    df["syllable_count_1"] = df["text_1"].apply(textstat.syllable_count)
    df["syllable_count_2"] = df["text_2"].apply(textstat.syllable_count)
    df["tfidf_cosine_sim"] = compute_tfidf_cosine(df["text_1"].tolist(), df["text_2"].tolist())

    X_simple = df[[ 
        "len_1", "len_2", "len_diff",
        "wordcount_1", "wordcount_2", "wordcount_diff",
        "readability_1", "readability_2",
        "avg_sentence_length_1", "avg_sentence_length_2",
        "syllable_count_1", "syllable_count_2",
        "tfidf_cosine_sim"
    ]].to_numpy()

    X_emb = np.concatenate([emb_1, emb_2, emb_1 - emb_2, emb_1 * emb_2], axis=1)

    return np.concatenate([X_emb, X_simple], axis=1)

def build_and_save_model():
    # === Labels ===
    y = (df["real_text_id"] == 2).astype(int) 

    # === Sentence Embeddings ===
    print("Generating sentence embeddings...")
    transformer = SentenceTransformer("all-mpnet-base-v2")
    emb_1 = transformer.encode(df["text_1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
    emb_2 = transformer.encode(df["text_2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    # Combine embeddings with features
    X_combined = extract_features(df, emb_1, emb_2)

    param_dist = {
        "n_estimators": randint(0, 1000),         # number of trees
        "max_depth": [None] + list(range(10, 51, 10)),  # deeper trees
        "min_samples_split": randint(2, 11),        # split threshold
        "min_samples_leaf": randint(1, 11),         # min leaf size
        "max_features": ["sqrt", "log2", None],     # how features are chosen
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=1000,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=42
    )

    print("🔍 Running RandomizedSearchCV...")
    search.fit(X_combined, y)

    print("✅ Best parameters found:", search.best_params_)
    print("📊 Best cross-validation accuracy:", search.best_score_)

    # Save the best model
    joblib.dump(search.best_estimator_, "rf_model.joblib")
    print("✅ Best model saved as rf_model.joblib")

if __name__ == "__main__":
    # Load text1 and text2 into DataFrame
    df[["text_1", "text_2"]] = df.apply(load_texts, axis=1)
    print("Loaded all text files!")
    build_and_save_model()