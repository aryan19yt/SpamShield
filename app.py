import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.result-spam {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    background-color: rgba(255,0,0,0.15);
}

.result-safe {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    background-color: rgba(0,255,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# NLP Preprocessing
# -------------------------

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()

    words = word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    stop_words = set(stopwords.words('english'))

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# -------------------------
# Load Model
# -------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# UI
# -------------------------

st.markdown(
    '<div class="title">🛡️ SpamShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP Powered SMS & Email Spam Detection</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "✉️ Enter a message",
    height=180,
    placeholder="Type or paste a message here..."
)

col1, col2 = st.columns([1,1])

with col1:
    predict_btn = st.button("🔍 Analyze Message")

with col2:
    clear_btn = st.button("🗑️ Clear")

if predict_btn:

    transformed = transform_text(message)

    vector_input = vectorizer.transform([transformed])

    prediction = model.predict(vector_input)[0]

    st.divider()

    if prediction == 1:
        st.markdown(
            '<div class="result-spam">🚨 SPAM DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-safe">✅ SAFE MESSAGE</div>',
            unsafe_allow_html=True
        )

st.divider()

st.caption(
    "Built using NLTK, TF-IDF, Naive Bayes, Scikit-Learn and Streamlit"
)