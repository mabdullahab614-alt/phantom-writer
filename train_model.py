import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# =======================================================
# 1. GENERATE DUMMY TRAINING DATA
# =======================================================
# In a real setup, you would load a massive CSV file here.
# Let's mock a balanced dataset: 0 = Human, 1 = AI

data = {
    "text": [
        "I went to the store today. I bought apples. The weather was very nice.",
        "Furthermore, it is important to realize the testament of technology. Moreover, we must consider its impact.",
        "Honestly, I think life is a beautiful blend of ups and downs. You just gotta keep moving forward.",
        "In conclusion, the digital transformation presents a unique opportunity. Consequently, adaptation is critical.",
        "My dog loves running in the backyard. He chased a squirrel up a tree yesterday and barked for hours.",
        "Therefore, the systematic implementation of advanced algorithms inherently optimizes the corporate paradigm.",
        "Cooking breakfast is my favorite morning routine. Coffee makes everything better anyway.",
        "It is globally recognized that artificial intelligence represents a profound paradigm shift in modern industries."
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Human, 1 = AI
}

df = pd.DataFrame(data)

# =======================================================
# 2. FEATURE EXTRACTION FUNCTION
# =======================================================
def extract_features(text):
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    if len(sentences) == 0:
        return [0.0, 0.0, 0.0]
        
    word_counts = [len(s.split()) for s in sentences]
    
    # Feature 1: Burstiness (Sentence variance)
    burstiness = float(np.std(word_counts)) if len(word_counts) > 1 else 0.0
    
    # Feature 2: Vocabulary Variety (Type-Token Ratio)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    ttr = (len(unique_words) / len(words)) if len(words) > 0 else 0.0
    
    # Feature 3: AI Marker Word Frequency
    ai_markers = ["furthermore", "therefore", "moreover", "consequently", "in conclusion", "testament", "paradigm"]
    marker_count = sum(1 for w in ai_markers if w in text.lower())
    
    return [burstiness, ttr, marker_count]

# Process the dataset to extract numerical features
X = np.array([extract_features(t) for t in df["text"]])
y = df["label"].values

# =======================================================
# 3. TRAINING THE RANDOM FOREST MODEL
# =======================================================
print("Initializing model training pipeline...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained intelligence engine to a file
with open("detector_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Success: 'detector_model.pkl' has been generated and saved!")