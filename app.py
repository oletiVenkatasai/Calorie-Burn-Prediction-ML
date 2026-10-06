import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title='Calorie Burn Predictor',
    page_icon='🔥',
    layout='centered'
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-header p {
    color: #a0a0c0;
    font-size: 1.1rem;
}

.stat-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-card h3 {
    color: #ffd93d;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}

.stat-card p {
    color: #a0a0c0;
    font-size: 0.85rem;
    margin: 0;
}

.result-box {
    background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,217,61,0.15));
    border: 2px solid rgba(255,107,107,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-box .calories {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-box .label {
    color: #a0a0c0;
    font-size: 1rem;
    margin-top: -0.5rem;
}

.form-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(8px);
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #ff6b6b, #ffd93d) !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ff6b6b, #ffd93d) !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    margin-top: 1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,107,107,0.4) !important;
}

label {
    color: #d0d0e8 !important;
    font-weight: 500 !important;
}

h2, h3 {
    color: white !important;
}

.interpretation {
    background: rgba(107,203,119,0.12);
    border-left: 4px solid #6bcb77;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    color: #d0d0e8;
}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.markdown('''
<div class="main-header">
    <h1>🔥 Calorie Burn Predictor</h1>
    <p>AI-Powered Exercise Calorie Estimation using KNN Regression</p>
</div>
''', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-card"><h3>99.52%</h3><p>Model Accuracy (R²)</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><h3>15K+</h3><p>Training Samples</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><h3>KNN</h3><p>Algorithm Used</p></div>', unsafe_allow_html=True)

st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('### 📋 Enter Your Details')

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('⚧ Gender', ['Male', 'Female'], key='gender')
    age = st.slider('🎂 Age (years)', min_value=20, max_value=79, value=30, step=1, key='age')
    height = st.slider('📏 Height (cm)', min_value=132, max_value=217, value=170, step=1, key='height')
    weight = st.slider('⚖️ Weight (kg)', min_value=36, max_value=119, value=70, step=1, key='weight')

with col2:
    duration = st.slider('⏱️ Exercise Duration (min)', min_value=1, max_value=30, value=15, step=1, key='duration')
    heart_rate = st.slider('❤️ Heart Rate (bpm)', min_value=67, max_value=122, value=90, step=1, key='heart_rate')
    body_temp = st.slider('🌡️ Body Temperature (°C)', min_value=38.2, max_value=40.0, value=39.5, step=0.1, key='body_temp')

st.markdown('</div>', unsafe_allow_html=True)

if st.button('🔥 Predict Calories Burned', key='predict_btn'):
    gender_encoded = 1 if gender == 'Male' else 0

    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction = max(0, round(prediction, 1))

    st.markdown(f'''
    <div class="result-box">
        <div class="calories">{prediction} kcal</div>
        <div class="label">Estimated Calories Burned</div>
    </div>
    ''', unsafe_allow_html=True)

    if prediction < 50:
        level = "🟡 Light Activity"
        tip = "Consider increasing workout duration or intensity for better results."
    elif prediction < 150:
        level = "🟠 Moderate Activity"
        tip = "Good effort! You're in a solid calorie-burning zone."
    elif prediction < 250:
        level = "🔴 High Intensity"
        tip = "Excellent workout! Make sure to stay hydrated and recover well."
    else:
        level = "💥 Very High Intensity"
        tip = "Outstanding performance! Elite-level calorie burn."

    st.markdown(f'''
    <div class="interpretation">
        <strong>{level}</strong><br>
        {tip}
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("📊 Input Summary"):
        import pandas as pd
        summary = pd.DataFrame({
            'Feature': ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart Rate', 'Body Temp'],
            'Value': [gender, f'{age} yrs', f'{height} cm', f'{weight} kg',
                      f'{duration} min', f'{heart_rate} bpm', f'{body_temp} °C']
        })
        st.dataframe(summary, use_container_width=True)

st.markdown('---')
st.markdown('<p style="text-align:center; color:#606080; font-size:0.8rem;">Built with ❤️ using Streamlit & Scikit-Learn | KNN Regression Model</p>', unsafe_allow_html=True)
