from pathlib import Path
import textstat
import pandas as pd
import numpy as np
from tqdm import tqdm
import build_model as bm
from sentence_transformers import SentenceTransformer
import joblib

# Load models
model = joblib.load("rf_model.joblib")  # Classifier
transformer = SentenceTransformer("all-mpnet-base-v2")

# === Load Test Data ===
test_dir = Path("./data/test")
test_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
print(f"Found {len(test_ids)} test samples.")

test_df = pd.DataFrame({"id": [int(i.split("_")[1]) for i in test_ids]})
test_df = test_df.sort_values("id").reset_index(drop=True)

# === Load Texts ===
def load_test_texts(row):
    folder = test_dir / bm.id_to_folder_name(row["id"])
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

# === Combine ===
X_test_combined = bm.extract_features(test_df, emb_test_1, emb_test_2)

# === Predict ===
y_pred = model.predict(X_test_combined)

# === Map back to real_text_id (1 or 2)
test_df["real_text_id"] = y_pred + 1  # Because label=1 → file_2 is real → real_text_id = 2

# === Save submission
submission = test_df[["id", "real_text_id"]]
submission.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv")
