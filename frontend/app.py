"""
app.py - Streamlit Frontend for Exam Anxiety Detector
======================================================
Soft Minimalism UI — Vibrant Purple/Indigo background with
Vibrant Green accents. Floating cards, clean white typography,
and visual anxiety-level indicators with emoji feedback.
"""

import streamlit as st
import requests
import time
import random

# ─── Configuration ────────────────────────────────────────────────────────────

API_URL       = "http://localhost:8000"
PREDICT_URL   = f"{API_URL}/predict"
HEALTH_URL    = f"{API_URL}/health"

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

  /* ── Root palette ── */
  :root {
    --bg-deep:       #0f0a1e;
    --bg-mid:        #1a1035;
    --bg-card:       rgba(30, 20, 60, 0.75);
    --border-soft:   rgba(139, 92, 246, 0.25);
    --border-green:  rgba(52, 211, 153, 0.45);

    --purple-light:  #a78bfa;
    --purple-mid:    #7c3aed;
    --indigo:        #4f46e5;

    --green-main:    #10b981;
    --green-bright:  #34d399;
    --green-glow:    rgba(52, 211, 153, 0.3);

    --text-white:    #f5f3ff;
    --text-dim:      rgba(245, 243, 255, 0.6);
    --text-muted:    rgba(245, 243, 255, 0.38);

    --low-color:     #34d399;
    --mod-color:     #fbbf24;
    --high-color:    #f87171;

    --shadow-card:   0 8px 32px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(139, 92, 246, 0.15);
    --shadow-green:  0 0 24px rgba(52, 211, 153, 0.25);
  }

  /* ── Base ── */
  html, body, .stApp {
    background: linear-gradient(135deg, #0f0a1e 0%, #1a1035 45%, #13062e 100%) !important;
    background-attachment: fixed !important;
    color: var(--text-white) !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* Subtle radial orbs */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 60% 50% at 15% 10%, rgba(124, 58, 237, 0.18) 0%, transparent 60%),
      radial-gradient(ellipse 50% 40% at 85% 85%, rgba(16, 185, 129, 0.10) 0%, transparent 55%),
      radial-gradient(ellipse 40% 35% at 70% 15%, rgba(79, 70, 229, 0.12) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  /* ── Layout container ── */
  .main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 780px !important;
  }

  /* ── All markdown text ── */
  .stMarkdown, .stMarkdown p, .stMarkdown li, label, .stTextArea label {
    color: var(--text-white) !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* ── Floating card base ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 22px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: var(--shadow-card);
    padding: 2.2rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.5), transparent);
  }

  /* ── Hero header ── */
  .hero-card {
    text-align: center;
    padding: 3.2rem 2rem 2.6rem;
    background: linear-gradient(145deg, rgba(79, 70, 229, 0.2), rgba(30, 20, 60, 0.7));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 26px;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(20px);
    margin-bottom: 2.2rem;
  }
  .hero-card .badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid var(--border-green);
    color: var(--green-bright) !important;
    border-radius: 50px;
    padding: 0.28rem 1.1rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
  }
  .hero-card h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin: 0 0 0.7rem !important;
    background: linear-gradient(135deg, #f5f3ff 30%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-card p {
    font-size: 1.05rem !important;
    color: var(--text-dim) !important;
    margin: 0 !important;
    font-weight: 300;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--purple-light) !important;
    margin-bottom: 0.7rem;
  }

  /* ── Text area ── */
  .stTextArea textarea {
    background: rgba(15, 10, 30, 0.6) !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: 16px !important;
    color: var(--text-white) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 1.1rem 1.2rem !important;
    caret-color: var(--green-bright);
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    resize: vertical !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--green-main) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
    outline: none !important;
  }
  .stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
  }
  /* Hide default Streamlit label */
  .stTextArea label { display: none !important; }

  /* ── Primary "Detect Anxiety" button ── */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    box-shadow: 0 8px 28px rgba(52, 211, 153, 0.5) !important;
    transform: translateY(-2px) !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
  }

  /* ── Secondary / sample text buttons ── */
  div[data-testid="stButton"] > button {
    background: rgba(139, 92, 246, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    color: var(--purple-light) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: rgba(139, 92, 246, 0.22) !important;
    border-color: rgba(167, 139, 250, 0.55) !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
  }

  /* ── Result anxiety card ── */
  .result-card {
    border-radius: 22px;
    padding: 2.4rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 1.6rem 0;
    animation: slideInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(16px);
  }
  .result-card.low {
    background: linear-gradient(160deg, rgba(16,185,129,0.12) 0%, rgba(30,20,60,0.7) 100%);
    border: 1.5px solid rgba(52, 211, 153, 0.45);
    box-shadow: 0 0 30px rgba(52, 211, 153, 0.12), var(--shadow-card);
  }
  .result-card.moderate {
    background: linear-gradient(160deg, rgba(251,191,36,0.10) 0%, rgba(30,20,60,0.7) 100%);
    border: 1.5px solid rgba(251, 191, 36, 0.4);
    box-shadow: 0 0 30px rgba(251, 191, 36, 0.10), var(--shadow-card);
  }
  .result-card.high {
    background: linear-gradient(160deg, rgba(248,113,113,0.12) 0%, rgba(30,20,60,0.7) 100%);
    border: 1.5px solid rgba(248, 113, 113, 0.4);
    box-shadow: 0 0 30px rgba(248, 113, 113, 0.12), var(--shadow-card);
  }
  .result-emoji {
    font-size: 4.5rem;
    line-height: 1;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3));
  }
  .result-level {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.35rem !important;
    color: var(--text-white) !important;
  }
  .result-confidence {
    font-size: 0.9rem;
    color: var(--text-dim) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-weight: 500;
  }

  /* ── Status pill ── */
  .pill {
    display: inline-block;
    border-radius: 50px;
    padding: 0.22rem 1rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }
  .pill-low      { background: rgba(52,211,153,0.15); color: #34d399 !important; border: 1px solid rgba(52,211,153,0.4); }
  .pill-moderate { background: rgba(251,191,36,0.12); color: #fbbf24 !important; border: 1px solid rgba(251,191,36,0.35); }
  .pill-high     { background: rgba(248,113,113,0.12); color: #f87171 !important; border: 1px solid rgba(248,113,113,0.35); }

  /* ── Progress bars ── */
  .stProgress > div > div > div > div {
    border-radius: 8px !important;
    transition: width 0.6s ease !important;
  }
  .stProgress { border-radius: 8px !important; overflow: hidden !important; }

  /* ── Tips box ── */
  .tip-item {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 1rem 1.1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.18);
    border-left: 3px solid var(--green-main);
    border-radius: 12px;
    margin-bottom: 0.75rem;
    font-size: 0.97rem;
    line-height: 1.6;
    color: var(--text-white) !important;
    transition: background 0.2s ease, transform 0.2s ease;
  }
  .tip-item:hover {
    background: rgba(255,255,255,0.055);
    transform: translateX(4px);
  }

  /* ── Disclaimer box ── */
  .disclaimer-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: rgba(251,191,36,0.85) !important;
    line-height: 1.55;
    margin-top: 1.4rem;
  }

  /* ── Hide sidebar toggle button ── */
  [data-testid="collapsedControl"] { display: none !important; }
  [data-testid="stSidebar"]        { display: none !important; }

  /* ── Character counter ── */
  .char-count {
    text-align: right;
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    margin-top: -0.4rem;
    margin-bottom: 0.6rem;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.8rem;
    border-top: 1px solid rgba(139,92,246,0.15);
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    line-height: 1.8;
  }

  /* ── Expander ── */
  details summary {
    color: var(--purple-light) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
  }

  /* ── Spinner text override ── */
  .stSpinner > div { color: var(--green-bright) !important; }

  /* ── Animations ── */
  @keyframes slideInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  .fade-in { animation: fadeIn 0.6s ease; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "Low":      "#34d399",
    "Moderate": "#fbbf24",
    "High":     "#f87171",
}
LEVEL_EMOJIS = {
    "Low":      "😊",
    "Moderate": "😐",
    "High":     "😟",
}
LEVEL_CSS = {
    "Low": "low",
    "Moderate": "moderate",
    "High": "high",
}
PILL_CSS = {
    "Low": "pill-low",
    "Moderate": "pill-moderate",
    "High": "pill-high",
}

SAMPLE_TEXTS = {
    "😌 Calm":        "I feel well-prepared for my exam tomorrow. I've studied all the major topics consistently and I'm confident in what I know. I just need a good night's sleep.",
    "😟 Restless":    "I'm a bit worried about the exam. There are a few topics I haven't fully covered. I keep second-guessing myself even on things I know well. I hope I do okay.",
    "😰 Overwhelmed": "I can't sleep. My hands are shaking whenever I open my textbook. I feel like no matter how much I study it'll never be enough. I'm terrified I'm going to fail.",
}




# ─── Hero Header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-card">
  <div class="badge">AI-Powered Mental Wellness</div>
  <h1>Exam Anxiety Detector</h1>
  <p>Share how you're feeling before your exam — our model will assess your anxiety level and offer personalised tips.</p>
</div>
""", unsafe_allow_html=True)


# ─── Input Card ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Your Thoughts</div>', unsafe_allow_html=True)

# Pre-load sample text via session state
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

user_text = st.text_area(
    label="thoughts",
    value=st.session_state.prefill,
    height=185,
    placeholder='Enter your exam-related thoughts or feelings…  e.g. I\'ve been studying for weeks but I still feel unprepared. Every time I look at the syllabus I feel overwhelmed.',
    key="input_area",
)

# Character / word count
if user_text:
    st.markdown(
        f'<div class="char-count">{len(user_text)} chars · {len(user_text.split())} words</div>',
        unsafe_allow_html=True,
    )

# ── Sample text shortcuts ──────────────────────────────────────────────────────

with st.expander("✦ Try a sample text"):
    cols = st.columns(3)
    for idx, (label, text) in enumerate(SAMPLE_TEXTS.items()):
        with cols[idx]:
            if st.button(label, use_container_width=True, key=f"sample_{idx}"):
                st.session_state.prefill = text
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Detect button ─────────────────────────────────────────────────────────────

detect_clicked = st.button(
    "🔍  Detect Anxiety",
    type="primary",
    use_container_width=True,
    disabled=not bool(user_text and user_text.strip()),
)


# ─── Prediction & Results ─────────────────────────────────────────────────────

if detect_clicked and user_text and user_text.strip():
    with st.spinner("Analyzing emotional patterns..."):
        time.sleep(0.6)                            # brief dramatic pause
        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": user_text.strip()},
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            st.error("🔴 **Cannot connect to backend.** Make sure FastAPI is running on port 8000.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⏱️ **Request timed out.** The model took too long. Please try again.")
            st.stop()

    if response.status_code != 200:
        st.error(f"❌ Backend error {response.status_code}: {response.text}")
        st.stop()

    data             = response.json()
    level            = data["anxiety_level"]          # "Low" | "Moderate" | "High"
    emoji            = LEVEL_EMOJIS.get(level, data.get("emoji", ""))
    confidence       = data["confidence"]
    confidence_scores= data["confidence_scores"]      # {Low, Moderate, High: float}
    tips             = data["tips"]
    disclaimer       = data["disclaimer"]

    color    = LEVEL_COLORS[level]
    card_cls = LEVEL_CSS[level]
    pill_cls = PILL_CSS[level]

    # ── Result card ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card {card_cls}">
      <div class="pill {pill_cls}">{level} Anxiety Detected</div>
      <div class="result-emoji">{emoji}</div>
      <div class="result-level" style="color:{color};">{level} Anxiety State</div>
      <div class="result-confidence">Confidence · {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Confidence Breakdown</div>', unsafe_allow_html=True)

    ordered_levels = ["Low", "Moderate", "High"]
    level_bar_colors = {
        "Low":      "#34d399",
        "Moderate": "#fbbf24",
        "High":     "#f87171",
    }
    for lv in ordered_levels:
        score = confidence_scores.get(lv, 0.0)
        bar_color = level_bar_colors[lv]
        col_name, col_bar = st.columns([1, 4])
        with col_name:
            st.markdown(
                f"<span style='color:{bar_color}; font-weight:600; font-size:0.95rem;'>{lv}</span>",
                unsafe_allow_html=True,
            )
        with col_bar:
            st.progress(round(score, 4))

    # ── Tips box ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Personalised Tips</div>', unsafe_allow_html=True)

    tips_to_show = tips[:5]                            # max 5 tips displayed
    tips_html = "".join(
        f'<div class="tip-item">{tip}</div>'
        for tip in tips_to_show
    )
    st.markdown(f'<div class="card fade-in">{tips_html}</div>', unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="disclaimer-box">{disclaimer}</div>',
        unsafe_allow_html=True,
    )

    # ── High-anxiety emergency block ──────────────────────────────────────────
    if level == "High":
        st.markdown("<br>", unsafe_allow_html=True)
        st.error(
            "🚨 **Immediate Support:** If you feel overwhelmed, please reach out to "
            "your institution's counseling service or a trusted person right away."
        )
        st.markdown(
            "**Crisis Helplines (India):**\n"
            "- 📞 **iCall** — 9152987821\n"
            "- 📞 **Vandrevala Foundation** — 1860-2662-345\n"
            "- 🌍 **International Crisis Text Line** — Text HOME to 741741"
        )

elif detect_clicked and not (user_text and user_text.strip()):
    st.warning("Please enter some text before clicking Detect.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
  🧠 Exam Anxiety Detector &nbsp;·&nbsp; AI-Powered Mental Wellness Tool<br>
  Non-diagnostic &amp; supportive only &nbsp;·&nbsp; Your data is never stored<br>
  &copy; 2026
</div>
""", unsafe_allow_html=True)
