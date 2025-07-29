# build_model.py

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
import joblib


transformer = SentenceTransformer("all-mpnet-base-v2") # Best score: 0.9251461988304094

# Load the ground truth
df = pd.read_csv("./data/train.csv")
print("Loaded train.csv with shape:", df.shape)

def load_texts(row):
    folder = Path("./data/train") / f"article_{row['id']:04d}"
    with open(folder / "file_1.txt", "r", encoding="utf-8") as f1:
        text1 = f1.read()
    with open(folder / "file_2.txt", "r", encoding="utf-8") as f2:
        text2 = f2.read()
    return pd.Series([text1, text2])

# === Feature Engineering ===
def extract_features(df, emb_1, emb_2):
    df["len_1"] = df["text_1"].str.len()
    df["len_2"] = df["text_2"].str.len()
    df["len_diff"] = df["len_1"] - df["len_2"]

    df["wordcount_1"] = df["text_1"].str.split().apply(len)
    df["wordcount_2"] = df["text_2"].str.split().apply(len)
    df["wordcount_diff"] = df["wordcount_1"] - df["wordcount_2"]

    df["cosine_sim"] = [cosine_similarity([a], [b])[0][0] for a, b in zip(emb_1, emb_2)]
    
    X_simple = df[[ 
        "len_1", "len_2", "len_diff",
        "wordcount_1", "wordcount_2", "wordcount_diff",
    ]].to_numpy()

    X_simple = np.hstack([X_simple, df[["cosine_sim"]].to_numpy()])

    X_emb = np.concatenate([
        # emb_1, 
        # emb_2, 
        emb_1 - emb_2, 
        emb_1 * emb_2
    ], axis=1)

    return np.concatenate([X_emb, X_simple], axis=1)

# === Train Model ===
def build_and_save_model(X_combined, y):
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None] + list(range(10, 80, 10)),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", None]
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="accuracy",
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # scores = cross_val_score(model, X_combined, y, cv=kf)
    # print("CV Accuracy with features + embeddings:", scores.mean())

    # model.fit(X_combined, y) # Use full train data now

    search.fit(X_combined, y)
    print("Best score:", search.best_score_)
    model = search.best_estimator_

    # # Save model
    # joblib.dump(model, "final_model.joblib")
    # print("✅ Model saved as final_model.joblib")

    return model

# === Predict Test ===
def predict_test(model):
    # === Load Test Data ===
    test_dir = Path("./data/test")
    test_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"Found {len(test_ids)} test samples.")

    test_df = pd.DataFrame({"id": [int(i.split("_")[1]) for i in test_ids]})
    test_df = test_df.sort_values("id").reset_index(drop=True)

    # === Load Texts ===
    def load_test_texts(row):
        folder = test_dir / f"article_{row['id']:04d}"
        with open(folder / "file_1.txt", "r", encoding="utf-8") as f1:
            text1 = f1.read()
        with open(folder / "file_2.txt", "r", encoding="utf-8") as f2:
            text2 = f2.read()
        return pd.Series([text1, text2])

    test_df[["text_1", "text_2"]] = test_df.apply(load_test_texts, axis=1)
    print("Loaded all test text files.")

    # === Generate Embeddings ===
    print("Generating test sentence embeddings...")
    emb_test_1 = transformer.encode(test_df["text_1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
    emb_test_2 = transformer.encode(test_df["text_2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    X_test_combined = extract_features(test_df, emb_test_1, emb_test_2)

    y_pred = model.predict(X_test_combined)

    # === Map back to real_text_id (1 or 2)
    test_df["real_text_id"] = y_pred + 1  # Because label=1 → file_2 is real → real_text_id = 2

    # === Save submission
    submission = test_df[["id", "real_text_id"]]
    submission.to_csv("submission.csv", index=False)
    print("✅ Saved submission.csv")

if __name__ == "__main__":
    # Load text1 and text2 into DataFrame
    df[["text_1", "text_2"]] = df.apply(load_texts, axis=1)
    print("Loaded all text files!")

    # Drop rows with empty file_1 or file_2
    empty = (df["text_1"].str.strip() == "") | (df["text_2"].str.strip() == "")
    print(f"Dropping {empty.sum()} rows with empty text files.")
    df = df[~empty].reset_index(drop=True)

    # Labels
    y = (df["real_text_id"] == 2).astype(int)  # 1 if file_2 is real
    
    # === Sentence Embeddings ===
    print("Generating sentence embeddings...")
    emb_1 = transformer.encode(df["text_1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
    emb_2 = transformer.encode(df["text_2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    # Combine embeddings with features
    X_combined = extract_features(df, emb_1, emb_2)

    model = build_and_save_model(X_combined, y)
    predict_test(model)