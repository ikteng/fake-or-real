import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

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

    X_simple = df[[ 
        "len_1", "len_2", "len_diff",
        "wordcount_1", "wordcount_2", "wordcount_diff",
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
        "n_estimators": randint(100, 500),         # number of trees
        "max_depth": [None] + list(range(1, 31, 10)),  # deeper trees
        "min_samples_split": randint(2, 8),        # split threshold
        "min_samples_leaf": randint(1, 5),         # min leaf size
        "max_features": ["sqrt", "log2", None],     # how features are chosen
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=100,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )

    print("🔍 Running RandomizedSearchCV...")
    search.fit(X_combined, y)

    print("✅ Best parameters found:", search.best_params_)
    print("📊 Best cross-validation accuracy:", search.best_score_)

    print("📊 Training Evaluation:")
    y_cv_pred = cross_val_predict(search.best_estimator_, X_combined, y, cv=cv, n_jobs=-1)
    print(classification_report(y, y_cv_pred))

    return search.best_estimator_

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
    transformer = SentenceTransformer("all-mpnet-base-v2")
    emb_test_1 = transformer.encode(test_df["text_1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
    emb_test_2 = transformer.encode(test_df["text_2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    X_test_combined = extract_features(test_df, emb_test_1, emb_test_2)

    # === Predict ===
    y_pred = model.predict(X_test_combined)

    # === Map back to real_text_id (1 or 2)
    test_df["real_text_id"] = y_pred + 1  # Because label=1 → file_2 is real → real_text_id = 2

    # === Save submission
    submission = test_df[["id", "real_text_id"]]
    # submission.to_csv("ST_submission1_noKFold.csv", index=False)
    submission.to_csv("ST_submission1_tunedCV.csv", index=False)
    print("\n📤 Sample Submission:\n", submission.head())

if __name__ == "__main__":
    # Load text1 and text2 into DataFrame
    df[["text_1", "text_2"]] = df.apply(load_texts, axis=1)
    print("Loaded all text files!")

    # Drop rows with empty file_1 or file_2
    empty_mask = (df["text_1"].str.strip() == "") | (df["text_2"].str.strip() == "")
    print(f"Dropping {empty_mask.sum()} rows with empty text files.")
    df = df[~empty_mask].reset_index(drop=True)

    model = build_and_save_model()

    predict_test(model)

# 📊 Best cross-validation accuracy: 0.9257309941520468
# 📊 Training Evaluation:
#               precision    recall  f1-score   support

#            0       0.89      0.89      0.89        45
#            1       0.90      0.90      0.90        48

#     accuracy                           0.89        93
#    macro avg       0.89      0.89      0.89        93
# weighted avg       0.89      0.89      0.89        93