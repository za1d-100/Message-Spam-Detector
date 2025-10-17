import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


# Streamlit Page Config
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered",
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Light orange background for the entire app */
    .stApp {
        background-color: #FFF3E0;
    }

    /* Main container background */
    .main .block-container {
        background-color: #FFF3E0;
        padding-top: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #E65100;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #FF9800;
        background-color: #FFFFFF;
        font-size: 16px;
        color: #333333;
    }

    .stTextArea textarea:focus {
        border: 2px solid #E65100;
        box-shadow: 0 0 8px rgba(230, 81, 0, 0.3);
    }

    /* Button styling - Dark orange */
    .stButton>button {
        background-color: #E65100;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 12em;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #BF360C;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(230, 81, 0, 0.3);
    }

    /* Result boxes */
    .result-box {
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
        border-radius: 10px;
        padding: 15px;
        border: 2px solid;
    }

    /* Sidebar background */
    .css-1d391kg {
        background-color: #FFE0B2;
    }

    /* Text color adjustments for better readability */
    .stMarkdown {
        color: #333333;
    }

    .stTextInput input {
        color: #333333 !important;
    }

    /* Caption styling */
    .stCaption {
        color: #666666;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# 🧠 Preprocessing Function
# -------------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# -------------------------------
# 📦 Load Model + Vectorizer
# -------------------------------
# Use relative paths (important for deployment!)
tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# -------------------------------
# 🎯 Streamlit App UI
# -------------------------------
st.markdown("<h1 class='main-title'>📩 Email / SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.write(
    "This simple AI app helps you detect whether a message is **Spam** or **Not Spam** using a trained Machine Learning model.")

input_sms = st.text_area("✉️ Enter your message below:")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        with st.spinner("Analyzing message..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown(
                "<div class='result-box' style='background-color:#FFEBEE; color:#C62828; border-color:#C62828;'>🚨 Spam Message Detected!</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#E8F5E8; color:#2E7D32; border-color:#2E7D32;'>✅ This message looks safe (Not Spam)</div>",
                unsafe_allow_html=True)

st.markdown("---")

st.caption("Developed by **Zaid Ansari 💻** | Powered by Machine Learning & Streamlit")


