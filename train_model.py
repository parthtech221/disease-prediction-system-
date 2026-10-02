import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ------------------- User-adjustable settings -------------------
CSV_FILENAME = "dataset.csv"
TARGET_COLUMN = "prognosis"
SAMPLE_FRAC = None  # e.g., 0.3 to sample 30% for quick testing
TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 12,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

MODEL_OUT = "model.pkl"
LABEL_ENCODER_OUT = "label_encoder.pkl"
FEATURES_OUT = "model_features.pkl"
ACCURACY_OUT = "model_accuracy.pkl"

# ------------------- Helper -------------------
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# ------------------- Main -------------------
def main():
    path = Path(CSV_FILENAME)
    if not path.exists():
        print(f"ERROR: {CSV_FILENAME} not found in {os.getcwd()}")
        return

    df = pd.read_csv(path)
    print("Loaded dataset:", df.shape)

    if SAMPLE_FRAC is not None and 0 < SAMPLE_FRAC < 1:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled {SAMPLE_FRAC*100:.1f}% of data ->", df.shape)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")
        return

    y_raw = df[TARGET_COLUMN]
    X_raw = df.drop(columns=[TARGET_COLUMN])

    # Fill missing values
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(num_cols):
        X_raw[num_cols] = X_raw[num_cols].fillna(X_raw[num_cols].median())
    if len(obj_cols):
        X_raw[obj_cols] = X_raw[obj_cols].fillna("Unknown")

    # Encode categorical features
    if obj_cols:
        X = pd.get_dummies(X_raw, columns=obj_cols, drop_first=True)
    else:
        X = X_raw.copy()

    X = downcast_df(X)
    print("Features after encoding:", X.shape)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.fillna("Unknown"))
    print(f"Number of classes: {len(le.classes_)} -> {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y))>1 else None
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    model = RandomForestClassifier(**RF_PARAMS)
    try:
        model.fit(X_train, y_train)
    except MemoryError:
        print("MemoryError -> retrying smaller model")
        model = RandomForestClassifier(n_estimators=10, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification report:\n", classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Save artifacts
    pickle.dump(model, open(MODEL_OUT, "wb"))
    pickle.dump(le, open(LABEL_ENCODER_OUT, "wb"))
    pickle.dump(list(X.columns), open(FEATURES_OUT, "wb"))
    pickle.dump(acc, open(ACCURACY_OUT, "wb"))

    print(f"\nSaved: {MODEL_OUT}, {LABEL_ENCODER_OUT}, {FEATURES_OUT}, {ACCURACY_OUT}")

if __name__ == "__main__":
    main()
