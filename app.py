from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class PredictionService:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and vectorizer"""
        try:
            # Load vectorizer
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load all trained models
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f != 'vectorizer.pkl']
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                with open(f'models/{model_file}', 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            print(f"Loaded {len(self.models)} models successfully!")
            
        except FileNotFoundError:
            print("Models not found. Please run train_models.py first.")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    def predict(self, text, model_type, algorithm):
        """Make prediction using specified model"""
        if self.vectorizer is None:
            return {"error": "Models not loaded"}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get model
        model_name = f"{model_type}_{algorithm}"
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0].tolist()
        
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "model": model_name
        }

# Initialize prediction service
prediction_service = PredictionService()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', '')
        algorithm = data.get('algorithm', '')
        
        if not text or not model_type or not algorithm:
            return jsonify({"error": "Missing required parameters"}), 400
        
        result = prediction_service.predict(text, model_type, algorithm)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models')
def get_available_models():
    """Get list of available models"""
    models = {}
    for model_name in prediction_service.models.keys():
        parts = model_name.split('_')
        if len(parts) >= 3:
            model_type = '_'.join(parts[:-1])
            algorithm = parts[-1]
            
            if model_type not in models:
                models[model_type] = []
            models[model_type].append(algorithm)
    
    return jsonify(models)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(prediction_service.models),
        "vectorizer_loaded": prediction_service.vectorizer is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
