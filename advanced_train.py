import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

def run_production_training():
    print("⚡ Loading DAIGT V4 Dataset...")
    try:
        df = pd.read_csv("train_v4_drcat_01.csv")
    except FileNotFoundError:
        print("❌ Error: 'train_v4_drcat_01.csv' missing.")
        return

    df = df.dropna(subset=['text', 'label'])
    
    # Strict Balance: Balanced slice to clear out background noise
    human_df = df[df['label'] == 0]
    ai_df = df[df['label'] == 1]
    
    sample_size = min(len(human_df), len(ai_df), 15000)
    print(f"⚖️ Dataset Balanced to {sample_size} rows per target class...")
    
    balanced_df = pd.concat([
        human_df.sample(n=sample_size, random_state=42),
        ai_df.sample(n=sample_size, random_state=42)
    ]).sample(frac=1, random_state=42) 

    X_data = np.array(balanced_df['text'].astype(str).tolist(), dtype=object)
    y_data = np.array(balanced_df['label'].astype(int).tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.15, random_state=42
    )
    
    print("💎 Blending Word & Character Sub-string matrices...")
    # Combines word tokens and character sub-combinations to ignore simple formatting bias
    vectorizer = FeatureUnion([
        ('word_dense', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
        ('char_dense', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=20000, sublinear_tf=True))
    ])
    
    # Using a slightly softer structural regularization penalty (C=1.2)
    full_model_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression(C=1.2, max_iter=1000, solver='saga', n_jobs=-1, random_state=42))
    ])
    
    print("🧠 Training Linear Core Engine...")
    full_model_pipeline.fit(X_train, y_train)
    
    predictions = full_model_pipeline.predict(X_test)
    print(f"\n📈 Tuned Base Accuracy: {round(accuracy_score(y_test, predictions) * 100, 2)}%")
    
    print("💾 Saving updated core to 'detector_model.pkl'...")
    with open("detector_model.pkl", "wb") as f:
        pickle.dump(full_model_pipeline, f)
    print("✅ Complete!")

if __name__ == "__main__":
    run_production_training()