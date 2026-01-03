from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

class PredictionService:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load the trained Pipeline models"""
        # Define the exact filenames we created in the notebook
        model_map = {
            'sms': 'models/sms_spam_model.pkl',
            'email': 'models/email_spam_model.pkl',
            'news': 'models/fake_news_model.pkl'
        }

        print("🔄 Loading Models...")
        for key, path in model_map.items():
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                    print(f"   ✅ {key.upper()} Model Loaded")
                except Exception as e:
                    print(f"   ❌ Error loading {key}: {e}")
            else:
                print(f"   ⚠️ File not found: {path}")

    def predict(self, text, model_type):
        """Make prediction using the specific pipeline"""
        
        # Default to SMS if model_type is weird
        if model_type not in self.models:
            return "Error: Model not found"
        
        model = self.models[model_type]
        
        # The pipeline handles Vectorization automatically!
        # We just pass the raw text in a list.
        prediction = model.predict([text])[0]
        
        # Convert 0/1 to Text
        if model_type == 'news':
            return "Fake News 🚨" if prediction == 1 else "Real News ✅"
        else:
            return "Spam 🚨" if prediction == 1 else "Safe ✅"

# Initialize prediction service
api = PredictionService()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get data from the webpage
        type_req = request.form.get('type')
        text_req = request.form.get('text')

        if not text_req:
            return jsonify({'prediction': "Please enter text!"})

        # Get Prediction
        result = api.predict(text_req, type_req)
        
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"})

if __name__ == '__main__':
    # Use the PORT environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)