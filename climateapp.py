import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import folium
from streamlit_folium import st_folium
import datetime

nltk.download('vader_lexicon')

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(page_title="Climate Change Modeling", layout="wide")
st.markdown("""
    <style>
        h1, h2, h3, h4, h5, h6 {
            color: #003366;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 16px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Climate Change Modeling Dashboard")

track = st.sidebar.radio("📌 Select Track", ["🗣️ Sentiment Analysis", "📈 Emission Forecasting"])

# -------------------- TRACK 1: SENTIMENT ANALYSIS --------------------
if track == "🗣️ Sentiment Analysis":
    st.header("🧠 Public Sentiment Analysis on NASA Climate Comments")
    try:
        df = pd.read_csv("climate_nasa.csv")
        df['text'] = df['text'].astype(str).fillna("")
        df['clean_text'] = df['text'].apply(lambda x: re.sub(r"http\S+|[^a-zA-Z\s]", "", x.lower()))

        sid = SentimentIntensityAnalyzer()
        df['score'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

        def classify(score):
            if score >= 0.05: return "Positive"
            elif score <= -0.05: return "Negative"
            else: return "Neutral"

        df['sentiment'] = df['score'].apply(classify)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Distribution")
            fig = plt.figure(figsize=(4, 3))
            sns.countplot(data=df, x='sentiment', palette='Set2')
            st.pyplot(fig)

        with col2:
            st.subheader("👍 Likes vs Sentiment")
            fig2 = plt.figure(figsize=(4, 3))
            sns.boxplot(data=df, x='sentiment', y='likesCount', palette='coolwarm')
            st.pyplot(fig2)

        st.subheader("📅 Sentiment Trend Over Time")
        monthly_avg = df.groupby('month')['score'].mean()
        fig3 = plt.figure(figsize=(6, 3))
        plt.plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o', color='#2a9d8f')
        plt.xticks(rotation=45)
        plt.ylabel("Average Sentiment Score")
        plt.grid(True)
        st.pyplot(fig3)

        st.subheader("📝 Sample Facebook Comments")
        st.dataframe(df[['text', 'sentiment', 'likesCount']].sample(5), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading climate_nasa.csv: {e}")

# -------------------- TRACK 2: EMISSION FORECASTING --------------------
elif track == "📈 Emission Forecasting":
    st.header("📉 CO₂ Emission Forecasting with Machine Learning")
    try:
        @st.cache_data
        def load_data():
            return pd.read_csv("train.csv"), pd.read_csv("test.csv")

        train, test = load_data()

        features = ['latitude', 'longitude', 'year', 'week_no']
        target = 'emission'

        train = train.dropna(subset=[target])
        train.fillna(train.median(numeric_only=True), inplace=True)
        test.fillna(test.median(numeric_only=True), inplace=True)

        X = train[features]
        y = train[target]
        X_test = test[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        st.subheader("📊 Model Evaluation Metrics")
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2_score(y_val, y_pred):.2f}")
        col2.metric("Mean Squared Error", f"{mean_squared_error(y_val, y_pred):.2f}")

        st.subheader("📈 Actual vs Predicted Emission")
        fig3 = plt.figure(figsize=(6, 4))
        plt.scatter(y_val, y_pred, alpha=0.4, color='mediumseagreen')
        plt.xlabel("Actual Emission")
        plt.ylabel("Predicted Emission")
        plt.title("Model Prediction Visualization")
        plt.grid(True)
        st.pyplot(fig3)

        st.subheader("🔍 Feature Importance (SHAP Analysis)")
        try:
            shap_sample = X.sample(n=200, random_state=42)
            explainer = shap.Explainer(model, shap_sample)
            shap_values = explainer(shap_sample)
            fig_shap = plt.figure()
            shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.warning(f"SHAP visualization error: {e}")

        st.subheader("🔁 Simulated Daily API Update")
        today = datetime.datetime.today().date()
        last_run = st.session_state.get("last_run", None)
        if last_run != today:
            st.session_state["last_run"] = today
            st.success("✅ Data updated from mock API today!")
        st.write("Last update:", st.session_state["last_run"])

        st.subheader("🔮 Forecast on Future Data")
        test['predicted_emission'] = model.predict(X_test_scaled)
        st.dataframe(test[['ID_LAT_LON_YEAR_WEEK', 'predicted_emission']].head(), use_container_width=True)

        csv = test[['ID_LAT_LON_YEAR_WEEK', 'predicted_emission']].to_csv(index=False)
        st.download_button("📥 Download Emission Predictions", data=csv, file_name="emission_predictions.csv", mime="text/csv")

        st.divider()

        # ---------------- Feature Sensitivity Simulator ----------------
        st.subheader("📊 Feature Sensitivity Simulator")
        with st.form("sensitivity_form"):
            selected_feature = st.selectbox("Select Feature to Test", features)
            steps = 20
            range_min = float(train[selected_feature].min())
            range_max = float(train[selected_feature].max())
            test_values = np.linspace(range_min, range_max, steps)

            default_input = {
                'latitude': float(train['latitude'].mean()),
                'longitude': float(train['longitude'].mean()),
                'year': int(train['year'].max()),
                'week_no': 25
            }

            user_input_copy = default_input.copy()
            sim_submit = st.form_submit_button("Run Sensitivity Simulation")

        if sim_submit:
            preds = []
            for val in test_values:
                user_input_copy[selected_feature] = val
                df_input = pd.DataFrame([user_input_copy])
                scaled_input = scaler.transform(df_input)
                pred = model.predict(scaled_input)[0]
                preds.append(pred)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(test_values, preds, marker='o')
            plt.xlabel(f"{selected_feature}")
            plt.ylabel("Predicted Emission")
            plt.title(f"Effect of {selected_feature} on Emission")
            plt.grid(True)
            st.pyplot(fig)

        # ---------------- Custom Prediction ----------------
        st.subheader("🧮 Predict Emission for Custom Input")
        with st.form("predict_form"):
            col1, col2, col3, col4 = st.columns(4)
            lat = col1.number_input("Latitude", value=float(train['latitude'].mean()))
            lon = col2.number_input("Longitude", value=float(train['longitude'].mean()))
            year = col3.number_input("Year", value=int(train['year'].max()))
            week = col4.slider("Week Number", 1, 52, 25)
            submit = st.form_submit_button("Predict")

        if submit:
            user_input = pd.DataFrame({'latitude': [lat], 'longitude': [lon], 'year': [year], 'week_no': [week]})
            scaled_input = scaler.transform(user_input)
            forecast = model.predict(scaled_input)[0]
            st.success(f"📊 Predicted Emission: {forecast:.2f} units")

    except Exception as e:
        st.error(f"❌ Error in emission forecasting: {e}")
