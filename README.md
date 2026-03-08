<div align="center">

# 🧠 Exam Anxiety Detector

### AI-Powered Mental Wellness Support for Students

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Analyze student text to detect exam-related anxiety levels using a deep neural network trained on TF-IDF features — wrapped in a calming, glassmorphic UI.*

</div>

---

## ✨ Features

- 🔍 **Real-time anxiety classification** — Low / Moderate / High from free-form student text
- 📊 **Confidence breakdown** — Per-class probability scores with visual progress bars
- 💡 **Personalized recommendations** — Contextual, anxiety-level-specific coping tips
- 🆘 **High-anxiety support** — Immediate helplines and emergency resources (India + Global)
- 🎨 **Stunning UI** — Ethereal glassmorphism dark theme with smooth animations
- ⚡ **REST API** — Clean FastAPI backend with Swagger docs at `/docs`
- 🔒 **Privacy-first** — No data stored; all inference is ephemeral

---

## 🏗️ Architecture

```
Exam Anxiety Detector
├── 🖥️  Streamlit Frontend  (port 8501)
│       └── Sends user text → FastAPI backend → renders results
└── ⚙️  FastAPI Backend     (port 8000)
        └── Loads TF-IDF vectorizer + Deep Neural Network
            └── Returns anxiety level, confidence scores & tips
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Custom CSS (Glassmorphism) |
| Backend API | FastAPI + Uvicorn |
| ML Model | PyTorch Deep Neural Network |
| Feature Extraction | Scikit-learn TF-IDF Vectorizer |
| Data Processing | Pandas |

---

## 📁 Project Structure

```
anxitey/
├── backend/
│   └── main.py             # FastAPI server — /predict, /health endpoints
├── frontend/
│   └── app.py              # Streamlit UI — glassmorphic design
├── model/
│   ├── bert_anxiety_model.pt    # Trained DNN model weights
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── data/
│   └── anxiety_dataset.csv      # Training dataset
├── scripts/
│   ├── preprocess.py       # Data cleaning and preprocessing
│   └── train.py            # Model training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Rajashree185/Detect_ExamAnxiety.git
cd Detect_ExamAnxiety
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model *(first time only)*

```bash
python scripts/preprocess.py
python scripts/train.py
```

> This generates `model/bert_anxiety_model.pt` and `model/tfidf_vectorizer.pkl`.

### 5. Run the Application

**Terminal 1 — Backend**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

**Then open:** [http://localhost:8501](http://localhost:8501)

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Predict anxiety level from text |
| `GET` | `/docs` | Interactive Swagger UI |

### `POST /predict`

**Request Body:**
```json
{
  "text": "I'm really worried about my calculus exam tomorrow..."
}
```

**Response:**
```json
{
  "anxiety_level": "Moderate",
  "emoji": "😟",
  "confidence": 0.8412,
  "confidence_scores": {
    "Low": 0.0891,
    "Moderate": 0.8412,
    "High": 0.0697
  },
  "tips": [
    "🧘 Try deep breathing: inhale for 4 seconds, hold for 4, exhale for 4.",
    "📝 Break your study material into smaller, manageable chunks."
  ],
  "disclaimer": "⚠️ This tool is for supportive purposes only..."
}
```

---

## 🧬 Model Details

| Property | Value |
|---|---|
| Architecture | Deep Neural Network (4-layer MLP) |
| Feature Extraction | TF-IDF (Scikit-learn) |
| Classification | 3-class: Low / Moderate / High |
| Regularization | BatchNorm + Dropout (0.3) |
| Framework | PyTorch |

The model uses a TF-IDF vectorizer to extract bag-of-words features from student text, which are then passed through a 4-layer fully connected neural network with BatchNorm and Dropout for regularization. The output is a softmax probability distribution over the three anxiety classes.

---

## 📊 Anxiety Levels

| Level | Emoji | Description |
|---|---|---|
| 🟢 **Low** | 😊 | Calm, confident, and prepared |
| 🟡 **Moderate** | 😟 | Some worry, manageable stress |
| 🔴 **High** | 😰 | Significant distress, needs support |

---

## 🛡️ Disclaimer

> ⚠️ This tool is for **supportive and educational purposes only**. It is **NOT a clinical or diagnostic assessment**. If you or someone you know is experiencing severe anxiety, please reach out to a qualified mental health professional.
>
> **Emergency Helplines:**
> - 🇮🇳 **iCall (India):** 9152987821
> - 🇮🇳 **Vandrevala Foundation:** 1860-2662-345
> - 🌍 **Crisis Text Line:** Text HOME to 741741

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ by [Rajashree185](https://github.com/Rajashree185)

*"One exam does not define your worth or your future."*

</div>
