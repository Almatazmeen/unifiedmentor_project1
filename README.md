# unifiedmentor_project1

# 🌍 Climate Change Modeling Dashboard

An interactive, dual-track Streamlit dashboard that visualizes **public sentiment** around climate issues and **predicts CO₂ emissions** using machine learning. This project combines **Natural Language Processing (NLP)**, **XGBoost regression**, **SHAP explainability**, and **interactive visualizations** to bring insights into climate change trends.

---

## 🚀 Features

### 📌 Two Main Tracks:
#### 1. 🗣️ Sentiment Analysis (Track 1)
- Analyzes public sentiment on climate change using Facebook comments.
- Uses NLTK VADER for sentiment classification: Positive, Neutral, Negative.
- Visualizes:
  - Sentiment distribution
  - Sentiment vs likes
  - Monthly sentiment trends
- View sample climate-related comments.

#### 2. 📈 Emission Forecasting (Track 2)
- Predicts CO₂ emissions using structured data (lat, lon, year, week_no).
- Built with:
  - XGBoost for regression
  - StandardScaler for normalization
- Evaluates with:
  - R² Score
  - Mean Squared Error
- Visual tools:
  - Actual vs Predicted scatter plot
  - SHAP feature importance (explainable AI)
  - Sensitivity simulator for feature effect
  - Custom input emission prediction
  - Downloadable prediction results

---

## 📁 File Structure

📦 Climate-Change-Dashboard  
├── climate_nasa.csv               # Facebook comments on climate  
├── train.csv                      # Training dataset for emission  
├── test.csv                       # Testing dataset for emission  
├── app.py                         # Main Streamlit app  
├── requirements.txt               # Dependencies  
└── README.md                      # This file  

---

## 🛠️ Technologies Used

- Streamlit – for interactive UI
- NLTK (VADER) – for sentiment analysis
- XGBoost – regression model for emission prediction
- SHAP – model explainability (feature importance)
- Matplotlib & Seaborn – data visualization
- Folium – geographic visualization (optional)
- Pandas, NumPy, Scikit-learn – data handling and ML

---
