import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK downloads (run once)
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# -----------------------------
# Load trained model
# -----------------------------
with open("Lr1_model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# YOUR CLEAN FUNCTION (UNCHANGED)
# -----------------------------
def clean(txt):
    regex = "[^a-zA-Z.]"
    doc = re.sub(regex, " ", txt)
    doc = doc.lower()

    tokens = nltk.word_tokenize(doc)

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(words) for words in filtered_tokens]

    return " ".join(lem_words)

# -----------------------------
# UI Visibility Fix (CSS ONLY)
# -----------------------------
st.markdown(
    """
    <style>
    /* Full background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1446317109212-0d94545661d0?w=900&auto=format&fit=crop&q=60");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Main content container */
    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 15px;
        color: white;
    }

    /* Title styling */
    h1 {
        color: #ff4b4b !important;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 10px black;
    }

    /* Subtext */
    p, label, span {
        color: #f0f0f0 !important;
        font-size: 16px;
    }

    /* Text area */
    textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: black !important;
        border-radius: 10px;
        font-size: 16px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
    }

    /* Success, warning, error boxes */
    .stAlert {
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# Fix caption visibility
# -----------------------------
st.markdown(
    """
    <style>
    /* Caption text (st.caption) */
    .stCaption {
        color: #ffffff !important;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
        text-shadow: 1px 1px 6px black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Suicide Detection", page_icon="🚨")
st.title("🧠 Suicide or Not-Suicide Detection")
st.write("Binary classification: Suicide vs Non-Suicide")

statement = st.text_area("Enter the statement", height=120)

# Explicit intent keywords (hard override)
HIGH_RISK_PHRASES = [
    "kill myself",
    "end my life",
    "want to die",
    "commit suicide",
    "take my life"
]

if st.button("Detect"):
    if statement.strip() == "":
        st.warning("⚠️ Please enter a valid statement.")
    else:
        raw_text = statement.lower()

        # Rule-based override (VERY IMPORTANT)
        if any(phrase in raw_text for phrase in HIGH_RISK_PHRASES):
            st.error("🚨 Suicide Detected (Explicit Intent)")
            st.info("⚠️ Please seek professional mental health support.")

        else:
            # Clean text EXACTLY like training
            cleaned_text = clean(statement)

            # Predict (MODEL EXPECTS CLEAN TEXT)
            prob = model.predict_proba([cleaned_text])[0][1]

            # Threshold tuned for suicide detection (recall > precision)
            THRESHOLD = 0.50

            if prob >= THRESHOLD:
                st.error(f"🚨 Suicide Detected ({prob*100:.2f}%)")
            else:
                st.success(f"✅ Non-Suicide ({(1 - prob)*100:.2f}%)")

            st.caption("Model: Logistic Regression | Input: Cleaned Text")

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Educational use only. Not a medical or psychological diagnosis."
)