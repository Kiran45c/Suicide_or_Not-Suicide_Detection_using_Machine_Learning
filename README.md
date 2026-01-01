# 🧠 Suicide or Non-Suicide Detection System

A **Streamlit-based Machine Learning web application** that classifies user-provided text as **Suicide** or **Non-Suicide** using **Natural Language Processing (NLP)** and **Logistic Regression**.

> ⚠️ This project is intended for **educational and research purposes only** and is **not a medical diagnostic tool**.

---

## 📌 Project Overview

The **Suicide or Non-Suicide Detection System** analyzes text input and determines whether it indicates suicidal intent.  
It combines **machine learning** with **rule-based safety checks** to improve reliability and reduce false negatives.

The application is deployed using **Streamlit** and features a clean, user-friendly interface.

---

## 🚀 Key Features

- ✅ Binary classification: **Suicide vs Non-Suicide**
- 🧹 Text preprocessing using **NLTK**
- 📊 Feature extraction with **TF-IDF**
- 🤖 Classification using **Logistic Regression**
- ⚠️ Rule-based override for explicit suicidal phrases
- 🎨 Enhanced UI with background image and improved readability
- ☁️ Ready for **Streamlit Cloud deployment**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **NLTK**
- **NumPy**
- **Pandas**

---

## ⚙️ How the System Works

1. User enters a text statement
2. Text is cleaned (lowercasing, tokenization, stopword removal, lemmatization)
3. Explicit high-risk phrases are checked first (rule-based safety layer)
4. Cleaned text is passed into a trained ML pipeline
5. The model predicts **Suicide** or **Non-Suicide**
6. The result is displayed in the UI

---

## 📂 Project Structure

├── app.py # Streamlit application
├── Lr1_model.pkl # Trained ML pipeline (TF-IDF + Logistic Regression)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## ⚠️ Disclaimer

This application is for educational use only.
It should not be used for medical, psychological, or clinical decision-making.

If you or someone you know is struggling, please seek professional mental health support.

## 🌱 Future Improvements

- Multi-class classification (Depression / Distress / Suicide)

- Improved dataset balancing

- Model explainability (SHAP / feature importance)

- Confidence visualization instead of raw probabilities

- Faster preprocessing without NLTK downloads

  ## ⭐ Acknowledgements

- NLTK – Natural Language Processing

- Scikit-learn – Machine Learning

- Streamlit – Web Application Framework
