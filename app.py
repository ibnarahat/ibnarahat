import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    target_col = next((c for c in ['target', 'num', 'HeartDisease'] if c in df.columns), df.columns[-1])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if set(y.unique()) - {0, 1}:
        y = (y > 0).astype(int)
    return X, y

X, y = load_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
)
model.fit(X_scaled, y)

st.set_page_config(
    page_title="Heart Disease Risk Predictor 💖",
    page_icon="💖",
    layout="centered"
)

st.title("💖 Heart Disease Risk Predictor")
st.write("Enter your health details below to predict the likelihood of heart disease.")

user_input = {}
st.subheader("🧍 Patient Health Information")

for col in X.columns:
    if X[col].dtype in [int, float]:
        user_input[col] = st.number_input(
            f"{col.replace('_',' ').capitalize()}",
            float(X[col].min()), float(X[col].max()), float(X[col].mean())
        )
    else:
        user_input[col] = st.selectbox(f"{col}", sorted(X[col].unique()))

input_df = pd.DataFrame([user_input])

input_scaled = scaler.transform(input_df.select_dtypes(include=[np.number]))

if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Chance: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Chance: {probability*100:.2f}%)")

st.subheader("💡 Most Important Risk Factors")

importance_df = pd.DataFrame({
    "Feature": X.select_dtypes(include=[np.number]).columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature").head(10))

st.caption("This app uses a machine learning model to estimate risk based on medical features.")