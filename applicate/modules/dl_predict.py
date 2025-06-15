import streamlit as st
import numpy as np
import pandas as pd
from src.data.data_processing import load_data
import plotly.graph_objects as go

def run_dl_predict():
    st.title("Derin Öğrenme Tahmin")
    
    # Session state kontrolü
    if 'trained_models' not in st.session_state or 'Deep Learning' not in st.session_state['trained_models']:
        st.warning("Önce 'Derin Öğrenme Algoritmaları' sayfasında model eğitimi yapmalısınız!")
        return
    
    # Model ve veriyi yükle
    model_info = st.session_state['trained_models']['Deep Learning']
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Veriyi yükle
    data = load_data("data.csv")
    if data is None:
        st.error("Veri yüklenemedi!")
        return
    
    # Özellik isimlerini al
    features = data.drop('diagnosis', axis=1).columns.tolist()
    
    # Tahmin seçenekleri
    st.subheader("Tahmin Seçenekleri")
    prediction_type = st.radio(
        "Tahmin Yöntemi",
        ["Manuel Giriş", "Rastgele Örnek"],
        help="Tahmin yapmak için veri giriş yöntemini seçin"
    )
    
    def show_gauge(prob, prediction_text):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': f"Kötü Huylu Olasılığı ({prediction_text})"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 0.33], 'color': "lightgreen"},
                    {'range': [0.33, 0.67], 'color': "gold"},
                    {'range': [0.67, 1], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        st.plotly_chart(fig)
    
    if prediction_type == "Manuel Giriş":
        st.write("Özellik değerlerini girin:")
        feature_values = {}
        for feature in features:
            min_val = data[feature].min()
            max_val = data[feature].max()
            mean_val = data[feature].mean()
            feature_values[feature] = st.slider(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100)
            )
        if st.button("Tahmin Et"):
            X = pd.DataFrame([feature_values])
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0][0]
            malign_prob = float(prediction)
            prediction_text = "Kötü Huylu (Malignant)" if malign_prob > 0.5 else "İyi Huylu (Benign)"
            st.subheader("Tahmin Sonuçları")
            show_gauge(malign_prob, prediction_text)
            if malign_prob > 0.5:
                st.error(f"Tahmin: Kötü Huylu (Malignant)\nOlasılık: {malign_prob:.2%}")
            else:
                st.success(f"Tahmin: İyi Huylu (Benign)\nOlasılık: {1-malign_prob:.2%}")
    else:
        if st.button("Rastgele Örnek Seç"):
            random_sample = data.sample(n=1)
            X = random_sample.drop('diagnosis', axis=1)
            y_true = random_sample['diagnosis'].iloc[0]
            st.write("Seçilen Örneğin Özellik Değerleri:")
            for feature in features:
                st.write(f"{feature}: {X[feature].iloc[0]:.4f}")
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0][0]
            malign_prob = float(prediction)
            prediction_text = "Kötü Huylu (Malignant)" if malign_prob > 0.5 else "İyi Huylu (Benign)"
            st.subheader("Tahmin Sonuçları")
            show_gauge(malign_prob, prediction_text)
            if malign_prob > 0.5:
                st.error(f"Tahmin: Kötü Huylu (Malignant)\nOlasılık: {malign_prob:.2%}")
            else:
                st.success(f"Tahmin: İyi Huylu (Benign)\nOlasılık: {1-malign_prob:.2%}")
            st.write(f"Gerçek Değer: {y_true}") 