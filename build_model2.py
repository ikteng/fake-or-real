import os
import re
import warnings
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


TRAIN_DIR = "./data/train"
TRAIN_CSV = "./data/train.csv"
TEST_DIR = "./data/test"
OUTPUT_FILE ="bert_submission.csv"
MODEL_CHECKPOINT = "bert-base-uncased"
# MODEL_CHECKPOINT = "microsoft/deberta-v3-small"

# --- Data Loaders ---
def train_data_generator(data_dir, csv_path):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        folder_id = row["id"]
        real_text_id = row["real_text_id"]
        folder_path = os.path.join(data_dir, f"article_{folder_id:04d}")
        with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1, \
             open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2:
            yield {
                "id": folder_id,
                "text1": f1.read(),
                "text2": f2.read(),
                "label": int(real_text_id == 1)
            }

def test_data_generator(data_dir):
    for folder in sorted(f for f in os.listdir(data_dir) if re.match(r'article_\d+', f)):
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(data_dir, folder)
        with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1, \
             open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2:
            yield {
                "id": folder_id,
                "text1": f1.read(),
                "text2": f2.read()
            }

# --- Feature Extraction ---
def extract_mean_pooling_vector(text, tokenizer, model, max_len=512, stride=256):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length"
    )
    vectors = []
    with torch.no_grad():
        for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden_state.size())
            mean = (hidden_state * mask).sum(1) / mask.sum(1)
            vectors.append(mean.squeeze(0))
    return torch.stack(vectors).mean(dim=0).cpu()

def extract_features(dataset, tokenizer, model):
    features, ids = [], []
    for row in tqdm(dataset, desc="Extracting features"):
        vec1 = extract_mean_pooling_vector(row['text1'], tokenizer, model)
        vec2 = extract_mean_pooling_vector(row['text2'], tokenizer, model)
        # vec1 = model.encode(row['text1'], convert_to_tensor=True)
        # vec2 = model.encode(row['text2'], convert_to_tensor=True)
        diff, prod = vec1 - vec2, vec1 * vec2
        final_vec = torch.cat([vec1, vec2, diff, prod])
        features.append(final_vec.numpy())
        ids.append(row['id'])
    return np.array(features), ids

# --- Model Training ---
def train_model(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)
    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

    for train_idx, val_idx in kf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y[val_idx], preds, average="macro")
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-score"].append(f1)

    df_results = pd.DataFrame({k: [np.mean(v)] for k, v in metrics.items()}).round(4).T
    df_results.columns = ['Value']
    print("\n✅ Cross-validation results for RandomForest:")
    print(df_results)

    # Retrain on all data
    model.fit(X, y)
    return model

# --- Inference ---
def predict_and_save(model, X_test, test_ids, output_file):
    probs = model.predict_proba(X_test)[:, 1]
    submission = [{"id": pid, "real_text_id": 1 if prob >= 0.5 else 2} for pid, prob in zip(test_ids, probs)]
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(output_file, index=False)
    print("\n📤 Sample Submission:")
    print(submission_df.head())

if __name__ == "__main__":
    print("🚀 Loading datasets...")
    raw_datasets = DatasetDict({
        "train": Dataset.from_generator(lambda: train_data_generator(TRAIN_DIR, TRAIN_CSV)),
        "test": Dataset.from_generator(lambda: test_data_generator(TEST_DIR))
    })

    print("📚 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    bert_model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(device)
    bert_model.eval()
    
    model = SentenceTransformer("all-mpnet-base-v2")

    print("🔍 Extracting training features...")
    X_train_raw, train_ids = extract_features(raw_datasets["train"], tokenizer, bert_model)
    # X_train_raw, train_ids = extract_features(raw_datasets["train"], model)

    # y_train = np.array([ex["label"] for ex in raw_datasets["train"]])
    y_train = np.array(raw_datasets["train"]["label"])

    print("📉 Applying PCA...")
    pca_model = PCA(n_components=20)
    X_train = pca_model.fit_transform(X_train_raw)

    print("🧠 Training model...")
    model = train_model(X_train, y_train)

    print("🧪 Extracting test features...")
    X_test_raw, test_ids = extract_features(raw_datasets["test"], tokenizer, bert_model)
    X_test = pca_model.transform(X_test_raw)

    print("📈 Predicting and saving submission...")
    predict_and_save(model, X_test, test_ids, output_file = OUTPUT_FILE)


# Cross-validation results for RandomForest:
#             Value
# Accuracy   0.9474
# Precision  0.9561
# Recall     0.9444
# F1-score   0.9465