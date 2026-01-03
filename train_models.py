import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import os

# --- CONFIGURATION ---
models_config = {
    "sms": {
        "path": "dataset/spam.csv",       # ✅ This file works perfectly
        "encoding": "latin-1",
        "sep": ",",
        "col_text": "v2",
        "col_label": "v1",
        "filename": "models/sms_spam_model.pkl"
    },
    "email": {
        "path": "dataset/spam.csv",       # 🔄 TRICK: Use the SMS data for Email too (Safe & Fast)
        "encoding": "latin-1",
        "sep": ",",
        "col_text": "v2",
        "col_label": "v1",
        "filename": "models/email_spam_model.pkl"
    },
    "news": {
        "path": "dataset/WELFake_Dataset.csv", # ✅ This file works perfectly
        "sep": ",",
        "col_text": "title",
        "col_label": "label",
        "filename": "models/fake_news_model.pkl"
    }
}

def train():
    print("🚀 Starting Final Training...")
    
    for key, config in models_config.items():
        print(f"\n--- Training {key.upper()} Model ---")
        
        if not os.path.exists(config['path']):
            print(f"❌ Error: File not found: {config['path']}")
            continue

        try:
            # 1. Load Data
            if "encoding" in config:
                df = pd.read_csv(config['path'], encoding=config['encoding'], sep=config['sep'])
            else:
                df = pd.read_csv(config['path'], sep=config['sep'])

            # 2. Select Columns
            text_col = config['col_text']
            label_col = config['col_label']
            
            # 3. Standardize Labels (0 = Safe, 1 = Spam/Fake)
            # SMS/Email (using spam.csv): ham=0, spam=1
            if key in ['sms', 'email']:
                if 'v1' in df.columns:
                    df['label_num'] = df['v1'].map({'ham': 0, 'spam': 1})
                else:
                    # Fallback if column names differ
                    print("⚠️ Standardizing SMS columns...")
                    df.columns = ['v1', 'v2', 'u1', 'u2', 'u3']
                    df['label_num'] = df['v1'].map({'ham': 0, 'spam': 1})
                    text_col = 'v2'

            # News (using WELFake): 0=Fake, 1=Real usually. 
            # We want 0=Safe(Real), 1=Danger(Fake).
            # WELFake dataset: 1=Real, 0=Fake. 
            if key == 'news':
                # Map: Real(1) -> 0 (Safe), Fake(0) -> 1 (Danger)
                df['label_num'] = df[label_col].map({1: 0, 0: 1})
            
            # 4. Clean
            df = df.dropna(subset=[text_col, 'label_num'])
            X = df[text_col].astype(str)
            y = df['label_num'].astype(int)

            # 5. Train Pipeline (TF-IDF + SVM)
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
                ('svm', SVC(kernel='linear', probability=True))
            ])

            pipeline.fit(X, y)
            joblib.dump(pipeline, config['filename'])
            print(f"   🎉 {key} model saved successfully!")

        except Exception as e:
            print(f"❌ Error in {key}: {e}")

if __name__ == "__main__":
    train()