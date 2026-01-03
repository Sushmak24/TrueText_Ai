# 🤖 TrueText AI

AI-powered text classification system for detecting spam, fraud, and fake news using Flask and scikit-learn.

## 📋 Features

- **Multi-Model Classification**: Uses optimized ML models for different text types
- **Multiple Detection Types**:
  - **📱 SMS Spam Detection**: Classify SMS messages as spam or legitimate
  - **📧 Email Spam Detection**: Identify spam in email communications  
  - **📰 Fake News Detection**: Detect fake news and misinformation
- **Advanced ML Algorithms**:
  - LinearSVC for fast SMS/Email classification
  - Logistic Regression for news analysis
  - TF-IDF vectorization with optimized parameters
- **Modern Web Interface**: Clean, responsive UI with real-time analysis
- **📄 Document Upload**: Support for PDF, DOCX, and TXT file analysis
- **Dual Input Methods**: Type text directly or upload documents for analysis
- **RESTful API**: Clean Flask backend with JSON responses
- **Optimized Performance**: Fast training with data sampling and parallel processing
- **Risk Assessment**: Provides clear classification results with confidence
- **Structured Output**: JSON-formatted predictions with model metadata
- **Fast Training**: Optimized pipeline for quick model training

## 🏗️ Project Structure

```
TrueText_Ai/
├── dataset/                    # Training datasets
│   ├── spam.csv               # SMS/Email spam dataset
│   ├── emails.csv             # Email dataset
│   └── WELFake_Dataset.csv    # Fake news dataset
├── models/                    # Trained models directory
│   ├── sms_spam_model.pkl     # SMS spam classifier
│   ├── email_spam_model.pkl   # Email spam classifier
│   └── fake_news_model.pkl    # Fake news classifier
├── templates/
│   └── index.html             # Modern web interface
├── app.py                     # Flask web application
├── train_models.py            # Optimized model training script
├── train_models.ipynb         # Jupyter notebook for training
├── requirements.txt           # Python dependencies
├── Procfile                   # Heroku deployment config
└── README.md                 # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Basic understanding of machine learning concepts

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd TrueText_Ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train the models** (First time only)
   ```bash
   python train_models.py
   ```

2. **Start the development server**
   ```bash
   python app.py
   ```

3. **Access the application**
   - **🌐 Live Demo**: https://truetext-ai.onrender.com
   - **Local Web Interface**: http://localhost:5000
   - **API Root**: http://localhost:5000
   - **Health Check**: http://localhost:5000/health

## � Document Upload Feature

### Supported File Formats

The application supports direct document analysis for the following formats:

- **📄 PDF Files** (.pdf) - Extracts text from all pages
- **📝 Word Documents** (.docx) - Processes paragraph content  
- **📃 Text Files** (.txt) - Direct text file reading

### How to Use Document Upload

1. **Navigate to**: https://truetext-ai.onrender.com
2. **Select Model Type**: Choose SMS, Email, or News detection
3. **Switch to File Tab**: Click "Upload File" tab
4. **Upload Document**: Select your PDF, DOCX, or TXT file
5. **Analyze**: Click "Analyze Content" to process the document

### Technical Implementation

```python
# Document processing uses:
- PyPDF2 for PDF text extraction
- python-docx for Word document processing
- UTF-8 encoding for text files
- Automatic text cleaning and preprocessing
```

### Use Cases

- **Email Analysis**: Upload email files for spam detection
- **Document Verification**: Analyze news articles or reports
- **Bulk Processing**: Process multiple documents efficiently
- **Content Review**: Check documents for suspicious content

## �📖 API Usage

### Text Classification

Analyze text for spam, fraud, or fake news.

**Endpoint:** `POST /predict`

**Request Body:** (Form data)
```
type: sms | email | news
text: "Your text content here"
# OR upload file
file: [PDF, DOCX, TXT file]
```

**Response:**
```json
{
  "prediction": "Spam" | "Ham" | "Fake" | "Real",
  "model": "sms_spam_model.pkl",
  "confidence": 0.95,
  "processing_time": 0.123
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:5000/predict" \
     -F "type=sms" \
     -F "text=Free entry in 2 a wkly comp to win FA Cup"
```

### Python Example

```python
import requests

# Classify SMS message
data = {'type': 'sms', 'text': 'Free entry win prize'}
response = requests.post("http://localhost:5000/predict", data=data)
result = response.json()
print(f"Prediction: {result['prediction']}")

# Upload PDF file for analysis
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'type': 'news'}
    response = requests.post("http://localhost:5000/predict", files=files, data=data)
    result = response.json()
    print(f"Prediction: {result['prediction']}")
```

## 🔧 Model Training

### Fast Training Script

The optimized training script includes:

- **Data Sampling**: Limits large datasets to 10,000 samples for faster training
- **Parallel Processing**: Uses all CPU cores with `n_jobs=-1`
- **Optimized Algorithms**: LinearSVC for speed, Logistic Regression for accuracy
- **Reduced Features**: 3,000 TF-IDF features instead of 5,000 for faster processing
- **Progress Tracking**: Real-time training progress and timing

```bash
python train_models.py
```

**Expected Output:**
```
🚀 Starting Fast Training...

--- Training SMS Model ---
   📊 Sampled 5572 rows from 5572 total
   🔄 Training sms model...
   🎉 sms model saved in 2.34 seconds!

--- Training EMAIL Model ---
   📊 Sampled 5572 rows from 5572 total
   🔄 Training email model...
   🎉 email model saved in 2.12 seconds!

--- Training NEWS Model ---
   📊 Sampled 10000 rows from 72134 total
   🔄 Training news model...
   🎉 news model saved in 8.45 seconds!

✅ All models trained in 13.91 seconds!
```

### Jupyter Notebook Training

For interactive training and experimentation:

```bash
jupyter notebook train_models.ipynb
```

## 🎯 Detection Capabilities

### SMS Spam Detection
- **Features**: Message content analysis
- **Accuracy**: 97-99%
- **Common Patterns**: Free offers, prizes, urgent requests

### Email Spam Detection  
- **Features**: Email body content analysis
- **Accuracy**: 95-98%
- **Common Patterns**: Marketing, phishing, suspicious links

### Fake News Detection
- **Features**: News headline and content analysis
- **Accuracy**: 85-90%
- **Common Patterns**: Sensationalism, misinformation, clickbait

## 🛠️ Development

### Adding New Models

1. **Add dataset** to `dataset/` folder
2. **Update configuration** in `train_models.py`:
   ```python
   models_config = {
       "new_type": {
           "path": "dataset/new_data.csv",
           "col_text": "text_column",
           "col_label": "label_column",
           "filename": "models/new_model.pkl"
       }
   }
   ```
3. **Train models**: `python train_models.py`
4. **Update web interface** in `templates/index.html`

### Model Performance Optimization

- **LinearSVC**: Faster than SVC, good for high-dimensional data
- **Logistic Regression**: Better for large datasets, provides probabilities
- **TF-IDF Parameters**:
  - `max_features=3000`: Balance between speed and accuracy
  - `min_df=2`: Ignore rare words
  - `max_df=0.8`: Ignore too common words
  - `ngram_range=(1,2)`: Include bigrams for better context

## 📊 Performance Metrics

### Training Speed (Optimized)
- **SMS Model**: ~2 seconds
- **Email Model**: ~2 seconds  
- **News Model**: ~8 seconds (with 10K samples)
- **Total Training**: ~13 seconds

### Model Size
- **SMS Model**: ~2MB
- **Email Model**: ~2MB
- **News Model**: ~5MB

### API Response Time
- **Classification**: <100ms per request
- **Model Loading**: ~500ms (first request only)

## 🔒 Security Best Practices

- **Input Validation**: All text inputs are sanitized
- **File Upload Limits**: Restricted file types and sizes
- **Model Security**: Models are saved in binary format
- **API Rate Limiting**: Consider implementing for production

## 🐛 Troubleshooting

### "Models not found"
```bash
# Train models first
python train_models.py
```

### "Dataset not found"
```bash
# Ensure datasets are in the dataset/ folder
ls dataset/
```

### "Port already in use"
```bash
# Change port in app.py
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use different port
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📈 Model Performance

Typical accuracy rates on test datasets:
- **SMS Spam**: 97-99% accuracy
- **Email Spam**: 95-98% accuracy  
- **Fake News**: 85-90% accuracy

*Note: Performance varies based on dataset quality and training parameters.*

## 🚀 Deployment

### Render Deployment

**🌐 Live Application**: https://truetext-ai.onrender.com

1. **Install Render CLI**
2. **Login to Render**: `render login`
3. **Create app**: `render create`
4. **Deploy**: `git push render main`

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**: `heroku login`
3. **Create app**: `heroku create truetext-ai`
4. **Deploy**: `git push heroku main`

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For questions or issues:
- Check the interactive web interface at `http://localhost:5000`
- Review API documentation in the code
- Open an issue on the repository

---

**Built with:** Flask | scikit-learn | pandas | Python 3.8+
