"""
main.py - FastAPI Backend for Anxiety Prediction
==================================================
Serves the trained anxiety classification model via REST API.
Endpoints:
  GET  /         - API info
  GET  /health   - Health check
  POST /predict  - Predict anxiety level from text input
"""

import os
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch.nn.functional as F

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "bert_anxiety_model.pt")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

NUM_LABELS = 3
LABEL_NAMES = ["Low", "Moderate", "High"]

TIPS = {
    "Low": [
        "🌟 You're doing great! Keep up the positive mindset.",
        "📚 Continue your steady study habits — consistency is key.",
        "😊 Remember to take short breaks to stay refreshed.",
        "💪 Your confidence will carry you through. Trust your preparation!"
    ],
    "Moderate": [
        "🧘 Try deep breathing: inhale for 4 seconds, hold for 4, exhale for 4.",
        "📝 Break your study material into smaller, manageable chunks.",
        "🚶 Take a 10-minute walk to clear your mind when feeling tense.",
        "💬 Talk to a friend, family member, or counselor about how you're feeling.",
        "🎵 Listen to calming music or nature sounds while you take a break.",
        "⏰ Create a realistic study schedule to avoid last-minute cramming."
    ],
    "High": [
        "🆘 Please reach out to a counselor or mental health professional for support.",
        "📞 Talk to someone you trust — a teacher, parent, or friend. You are not alone.",
        "🧘‍♀️ Practice grounding: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
        "💤 Prioritize sleep and rest — your health matters more than any exam.",
        "🛑 Step away from studying if it's causing panic. Your well-being comes first.",
        "❤️ Remember: one exam does not define your worth or your future.",
        "🌿 Try progressive muscle relaxation to release physical tension.",
        "📱 Consider using a meditation app like Headspace or Calm for guided relaxation."
    ]
}

EMOJI_MAP = {
    "Low": "😊",
    "Moderate": "😟",
    "High": "😰"
}


# ─── Neural Network Model (must match train.py) ─────────────────────────────

class AnxietyClassifier(nn.Module):
    """Deep Neural Network for anxiety text classification."""

    def __init__(self, input_size, hidden_size, num_labels, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_size // 4, num_labels)
        )

    def forward(self, x):
        return self.network(x)


# ─── Request/Response Models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Student text input")

class PredictResponse(BaseModel):
    anxiety_level: str
    emoji: str
    confidence: float
    confidence_scores: dict
    tips: list[str]
    disclaimer: str


# ─── App Initialization ─────────────────────────────────────────────────────

app = FastAPI(
    title="Exam Anxiety Detector API",
    description="AI-powered system to detect and categorize exam-related anxiety from student text.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
vectorizer = None


# ─── Model Loading ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    """Load the trained model and TF-IDF vectorizer on startup."""
    global model, vectorizer

    print("=" * 50)
    print("  Loading Anxiety Detection Model...")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"  ERROR: Model or vectorizer not found!")
        print(f"  Model:      {MODEL_PATH} -> {'EXISTS' if os.path.exists(MODEL_PATH) else 'MISSING'}")
        print(f"  Vectorizer: {VECTORIZER_PATH} -> {'EXISTS' if os.path.exists(VECTORIZER_PATH) else 'MISSING'}")
        print("  Run 'python scripts/train.py' first to train the model.")
        print("=" * 50)
        return

    # Load vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"  Vectorizer loaded (vocab size: {len(vectorizer.vocabulary_)})")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model = AnxietyClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_labels=checkpoint["num_labels"],
        dropout=checkpoint.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_accuracy", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    print(f"  Model loaded successfully!")
    print(f"  Trained Epoch: {epoch}")
    print(f"  Validation Accuracy: {val_acc}")
    print("=" * 50)


# ─── Prediction Logic ───────────────────────────────────────────────────────

def predict_anxiety(text: str) -> dict:
    """Run inference on input text and return anxiety prediction."""
    device = next(model.parameters()).device

    # Vectorize the text
    text_features = vectorizer.transform([text])
    text_tensor = torch.FloatTensor(text_features.toarray()).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(text_tensor)
        probabilities = F.softmax(outputs, dim=1)

    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    # Build confidence scores dict
    confidence_scores = {
        LABEL_NAMES[i]: round(probabilities[0][i].item(), 4)
        for i in range(NUM_LABELS)
    }

    anxiety_level = LABEL_NAMES[predicted_class]

    return {
        "anxiety_level": anxiety_level,
        "emoji": EMOJI_MAP[anxiety_level],
        "confidence": round(confidence, 4),
        "confidence_scores": confidence_scores,
        "tips": TIPS[anxiety_level]
    }


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Exam Anxiety Detector API"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict the anxiety level from student text input.

    - **text**: Student's written text (reflection, feedback, or pre-exam thoughts)

    Returns anxiety level (Low/Moderate/High) with confidence scores and management tips.
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python scripts/train.py' first, then restart the server."
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    result = predict_anxiety(text)
    result["disclaimer"] = (
        "⚠️ This tool is for supportive purposes only and is NOT a clinical or diagnostic assessment. "
        "If you are experiencing severe anxiety, please reach out to a qualified mental health professional."
    )

    return result


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Exam Anxiety Detector API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
