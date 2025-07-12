# EDA.py
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import textstat

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

def EDA():
    # Compare lengths
    df["len_1"] = df["text_1"].str.len()
    df["len_2"] = df["text_2"].str.len()

    df["wordcount_1"] = df["text_1"].str.split().apply(len)
    df["wordcount_2"] = df["text_2"].str.split().apply(len)

    # Print summary stats
    print("\nCharacter length statistics:")
    print("Text 1 - mean:", df["len_1"].mean(), "min:", df["len_1"].min(), "max:", df["len_1"].max())
    print("Text 2 - mean:", df["len_2"].mean(), "min:", df["len_2"].min(), "max:", df["len_2"].max())

    print("\nWord count statistics:")
    print("Text 1 - mean:", df["wordcount_1"].mean(), "min:", df["wordcount_1"].min(), "max:", df["wordcount_1"].max())
    print("Text 2 - mean:", df["wordcount_2"].mean(), "min:", df["wordcount_2"].min(), "max:", df["wordcount_2"].max())

    # Difference in length
    df["len_diff"] = df["len_1"] - df["len_2"]
    df["wordcount_diff"] = df["wordcount_1"] - df["wordcount_2"]

    print("\nDifference in character length (text_1 - text_2):")
    print(df["len_diff"].describe())

    print("\nDifference in word count (text_1 - text_2):")
    print(df["wordcount_diff"].describe())

    # Check distributions
    plt.hist(df["len_1"], bins=30, alpha=0.5, label="Text 1")
    plt.hist(df["len_2"], bins=30, alpha=0.5, label="Text 2")
    plt.legend()
    plt.title("Character length distribution")
    plt.show()

    plt.hist(df["len_diff"], bins=30)
    plt.title("Text 1 - Text 2 length difference")
    plt.show()

    # Real vs Fake length comparison
    df["real_len"] = df.apply(lambda row: row["len_1"] if row["real_text_id"] == 1 else row["len_2"], axis=1)
    df["fake_len"] = df.apply(lambda row: row["len_2"] if row["real_text_id"] == 1 else row["len_1"], axis=1)

    # plt.hist(df["real_len"], bins=30, alpha=0.5, label="Real")
    plt.hist(df["fake_len"], bins=30, alpha=0.5, label="Fake")
    plt.legend()
    plt.title("Real vs Fake Length Distribution")
    plt.show()

    print("Average real length:", df["real_len"].mean())
    print("Average fake length:", df["fake_len"].mean())

    plt.hist(df["real_len"], bins=30, alpha=0.5, label="Real")
    plt.hist(df["fake_len"], bins=30, alpha=0.5, label="Fake")
    plt.legend()
    plt.title("Real vs Fake Length Distribution")
    plt.show()

    df["readability_1"] = df["text_1"].apply(textstat.flesch_reading_ease)
    df["readability_2"] = df["text_2"].apply(textstat.flesch_reading_ease)

    df["real_readability"] = df.apply(lambda row: row["readability_1"] if row["real_text_id"] == 1 else row["readability_2"], axis=1)
    df["fake_readability"] = df.apply(lambda row: row["readability_2"] if row["real_text_id"] == 1 else row["readability_1"], axis=1)

    print("Average Real Readability:", df["real_readability"].mean())
    print("Average Fake Readability:", df["fake_readability"].mean())

if __name__ == "__main__":
    # Load text1 and text2 into DataFrame
    df[["text_1", "text_2"]] = df.apply(load_texts, axis=1)
    print("Loaded all text files!")
    # EDA()
