from flask import Flask, render_template, request, jsonify
import joblib
import os
import PyPDF2
import docx

app = Flask(__name__)

class PredictionService:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        # Define the exact filenames
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
        if model_type not in self.models:
            return "Error: Model not found"
        
        model = self.models[model_type]
        prediction = model.predict([text])[0]
        
        if model_type == 'news':
            return "Fake News 🚨" if prediction == 1 else "Real News ✅"
        else:
            return "Spam 🚨" if prediction == 1 else "Safe ✅"

api = PredictionService()

def extract_text_from_file(file):
    """Helper function to read text from PDF, DOCX, or TXT"""
    filename = file.filename.lower()
    text = ""

    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        elif filename.endswith('.docx'):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + " "
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
    except Exception as e:
        return None
    
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        type_req = request.form.get('type')
        text_req = request.form.get('text')
        
        # Check if a file was uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                text_req = extracted_text
            else:
                return jsonify({'prediction': "Error: Could not read file."})

        if not text_req:
            return jsonify({'prediction': "Please enter text or upload a file!"})

        result = api.predict(text_req, type_req)
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)