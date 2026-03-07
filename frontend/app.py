"""
app.py - Streamlit Frontend for Exam Anxiety Detector
======================================================
Provides an intuitive UI for students to input text and
view their predicted anxiety level with visual indicators,
emojis, and anxiety-management tips.
"""

import streamlit as st
import requests
import time

# ─── Configuration ───────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS (Frontend Design Skill: Ethereal Glassmorphism) ─────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

    :root {
        --bg-color: #0d1117;
        --glass-bg: rgba(20, 25, 35, 0.45);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-highlight: rgba(255, 255, 255, 0.15);
        --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        
        --accent-glow: #58a6ff;
        
        --low-anxiety: #2ea043;
        --mod-anxiety: #d29922;
        --high-anxiety: #f85149;
    }

    /* Overall Background setup */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(88, 166, 255, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(138, 43, 226, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(20, 25, 35, 0.8) 0%, transparent 80%);
        background-attachment: fixed;
        color: var(--text-primary);
    }
    
    /* Protect Streamlit internal styling */
    .material-symbols-rounded, .material-icons, .icon {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }
    
    /* Targeted Typography overrides (avoids breaking Streamlit icons) */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: var(--text-primary) !important;
    }
    
    /* Apply color without breaking Streamlit font inheritance for icons */
    .stMarkdown p, .stMarkdown li {
        color: var(--text-primary);
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 760px;
        animation: fadeIn 1.2s ease-out;
    }

    /* Header styling - Glassmorphic */
    .main-header {
        text-align: center;
        padding: 3.5rem 2rem;
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        margin-bottom: 3rem;
        box-shadow: var(--glass-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--glass-highlight), transparent);
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #ffffff, #a5c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(88, 166, 255, 0.3);
    }
    .main-header p {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.15rem;
        font-weight: 300;
        letter-spacing: 0.02em;
        margin: 0;
        color: var(--text-secondary);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background: rgba(10, 15, 20, 0.6) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-glow) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2), inset 0 2px 10px rgba(0,0,0,0.5) !important;
    }
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.8) !important;
    }

    /* Button Styling */
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        backdrop-filter: blur(8px) !important;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: var(--glass-highlight) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(88,166,255,0.2) 0%, rgba(88,166,255,0.05) 100%) !important;
        border: 1px solid rgba(88, 166, 255, 0.4) !important;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.15) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(88,166,255,0.3) 0%, rgba(88,166,255,0.1) 100%) !important;
        border: 1px solid rgba(88, 166, 255, 0.6) !important;
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.25) !important;
    }

    /* Result cards */
    .anxiety-card {
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: var(--glass-shadow);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
        animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .anxiety-card h1 {
        font-size: 4rem !important;
        margin-bottom: 0.5rem;
        line-height: 1;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.3));
    }
    
    .anxiety-card h2 {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem;
    }

    .anxiety-card.low {
        border-top: 2px solid var(--low-anxiety);
        background: linear-gradient(180deg, rgba(46, 160, 67, 0.05) 0%, rgba(20, 25, 35, 0.4) 100%);
    }
    .anxiety-card.moderate {
        border-top: 2px solid var(--mod-anxiety);
        background: linear-gradient(180deg, rgba(210, 153, 34, 0.05) 0%, rgba(20, 25, 35, 0.4) 100%);
    }
    .anxiety-card.high {
        border-top: 2px solid var(--high-anxiety);
        background: linear-gradient(180deg, rgba(248, 81, 73, 0.05) 0%, rgba(20, 25, 35, 0.4) 100%);
    }

    /* Tip card */
    .tip-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        transition: transform 0.3s ease;
        border-left: 3px solid var(--glass-highlight);
        font-family: 'Outfit', sans-serif;
    }
    .tip-card:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateX(5px);
        border-left-color: var(--accent-glow);
    }

    /* Disclaimer card */
    .disclaimer-card {
        background: rgba(210, 153, 34, 0.05);
        border: 1px solid rgba(210, 153, 34, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 2rem;
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.7) !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-color: var(--accent-glow) !important;
        border-radius: 10px;
    }
    .stProgress {
        background: rgba(0,0,0,0.3) !important;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(10, 15, 20, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--glass-border) !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.4) !important;
        font-size: 0.85rem;
        margin-top: 4rem;
        padding-bottom: 1rem;
        border-top: 1px solid var(--glass-border);
        font-family: 'Outfit', sans-serif;
        padding-top: 2rem;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Subheaders styling explicitly applied */
    .stMarkdown h3 {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--glass-border);
        display: inline-block;
    }

</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>Exam Anxiety Detector</h1>
    <p>AI-powered ethereal mental wellness support</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✧ About")
    st.markdown(
        "This tool uses a **BERT-based AI model** to analyze text "
        "and detect exam-related anxiety levels, wrapped in a calming interface."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✧ Anxiety Levels")
    st.markdown("<span style='color:var(--low-anxiety)'>●</span> **Low** — Calm and confident", unsafe_allow_html=True)
    st.markdown("<span style='color:var(--mod-anxiety)'>●</span> **Moderate** — Some worry present", unsafe_allow_html=True)
    st.markdown("<span style='color:var(--high-anxiety)'>●</span> **High** — Significant distress", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✧ Privacy")
    st.markdown(
        "Your text is **not stored** anywhere. "
        "All analysis is done in real-time and data is "
        "discarded after processing."
    )


# ─── Main Input Section ─────────────────────────────────────────────────────

st.markdown("### Share Your Thoughts")
st.markdown(
    "Write about how you're feeling for your upcoming exam. "
    "Take a deep breath. We're here to understand."
)

user_text = st.text_area(
    "Your text input:",
    height=200,
    placeholder="Take a moment. What's on your mind regarding your upcoming exams? Are you feeling prepared or overwhelmed? Write freely...",
    label_visibility="collapsed"
)

# Character count
if user_text:
    st.caption(f"✧ {len(user_text)} characters • {len(user_text.split())} words")


# ─── Sample Texts ────────────────────────────────────────────────────────────

with st.expander("✧ Try a curated sample text"):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Calm State", use_container_width=True):
            st.session_state["sample_text"] = (
                "I feel well-prepared for my exam. I studied consistently "
                "and I'm confident in my understanding of the material."
            )
            st.rerun()

    with col2:
        if st.button("Restless", use_container_width=True):
            st.session_state["sample_text"] = (
                "I'm a bit worried about the exam. I studied most topics "
                "but I'm not sure if I covered everything. I hope I do okay."
            )
            st.rerun()

    with col3:
        if st.button("Overwhelmed", use_container_width=True):
            st.session_state["sample_text"] = (
                "I'm terrified about the exam. I can't sleep, my hands are "
                "shaking, and I feel like I'm going to fail no matter what."
            )
            st.rerun()

# Load sample text if selected
if "sample_text" in st.session_state:
    user_text = st.session_state.pop("sample_text")


# ─── Analysis Button ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button(
    "Analyze My Text ✧",
    type="primary",
    use_container_width=True,
    disabled=not user_text
)

if analyze_clicked and user_text:
    with st.spinner("Analyzing neural patterns..."):
        try:
            # Call the FastAPI backend
            response = requests.post(
                PREDICT_ENDPOINT,
                json={"text": user_text},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                anxiety_level = result["anxiety_level"]
                emoji = result["emoji"]
                confidence = result["confidence"]
                confidence_scores = result["confidence_scores"]
                tips = result["tips"]
                disclaimer = result["disclaimer"]

                # Small delay for effect
                time.sleep(0.8)

                # ─── Display Results ─────────────────────────────────

                # Anxiety level card
                card_class = anxiety_level.lower()
                st.markdown(f"""
                <div class="anxiety-card {card_class}">
                    <h1>{emoji}</h1>
                    <h2 style='font-family: "Playfair Display", serif;'>{anxiety_level} State Detected</h2>
                    <p style="font-size: 1.15rem; opacity: 0.7; letter-spacing: 1px; text-transform: uppercase;">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence breakdown
                st.markdown("### Confidence Breakdown")
                st.markdown("<br>", unsafe_allow_html=True)
                for level_name, score in confidence_scores.items():
                    col_label, col_bar = st.columns([1, 4])
                    with col_label:
                        color = "var(--low-anxiety)" if level_name == 'Low' else "var(--mod-anxiety)" if level_name == 'Moderate' else "var(--high-anxiety)"
                        st.markdown(f"<span style='color: {color}; font-weight: 500;'>{level_name}</span>", unsafe_allow_html=True)
                    with col_bar:
                        st.progress(score)

                # Tips section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Recommendations")
                for tip in tips:
                    st.markdown(f"""
                    <div class="tip-card">{tip}</div>
                    """, unsafe_allow_html=True)

                # Disclaimer
                st.markdown(f"""
                <div class="disclaimer-card">
                    ✧ {disclaimer}
                </div>
                """, unsafe_allow_html=True)

                # Additional resources for high anxiety
                if anxiety_level == "High":
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.error(
                        "🚨 **Immediate Support:** If you feel overwhelmed, "
                        "please contact your institution's counseling service."
                    )
                    st.markdown(
                        "**Helplines:**\n"
                        "- 🇮🇳 **iCall**: 9152987821\n"
                        "- 🇮🇳 **Vandrevala Foundation**: 1860-2662-345\n"
                        "- 🌍 **Crisis Text Line**: Text HOME to 741741"
                    )

            else:
                st.error(f"✧ Server Connection Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "✧ **Cannot connect to neural backend.** "
                "Ensure FastAPI is running."
            )
        except requests.exceptions.Timeout:
            st.error("✧ **Request timed out.** Network is under strain. Try again.")
        except Exception as e:
            st.error(f"✧ **Unexpected error:** {str(e)}")


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer-text">
    Ethereal Exam Anxiety Detector • A neural wellness experience<br>
    Non-diagnostic supportive tool • Your data is ephemeral<br>
    © 2026
</div>
""", unsafe_allow_html=True)

