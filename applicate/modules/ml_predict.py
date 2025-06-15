import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.data.data_processing import load_data
from models.ml.ml_models import load_ml_model, predict_ml
import streamlit.components.v1 as components

# Kullanılacak önemli özellikler
important_features = [
    'radius_mean', 'perimeter_mean', 'area_mean', 
    'concavity_mean', 'concave points_mean',
    'radius_worst', 'perimeter_worst', 'area_worst',
    'concavity_worst', 'concave points_worst'
]

def run_ml_predict():
    st.title("Meme Kanseri Veri Analizi ve Tahmini (ML)")
    st.header("Makine Öğrenmesi ile Meme Kanseri Tahmini")
    
    # Model kontrolü
    if 'trained_models' not in st.session_state:
        st.warning("Lütfen önce model eğitimini tamamlayın.")
        return

    # Model seçimi
    st.subheader("Model Seçimi")
    available_models = list(st.session_state['trained_models'].keys())
    selected_model = st.selectbox(
        "Tahmin yapmak istediğiniz modeli seçin:",
        options=available_models,
        index=0
    )

    st.subheader("Özellik Değerlerini Girin")
    data = load_data("data.csv")
    
    # Eğer rastgele örnek yüklenecekse, slider'lardan ÖNCE session_state'e atama yapılmalı
    if st.session_state.get("load_random", False):
        sample_idx = np.random.randint(0, len(data))
        sample = data.iloc[sample_idx]
        for feature in important_features:
            key_name = f"slider_{feature}"
            st.session_state[key_name] = float(sample[feature])
        st.session_state["load_random"] = False

    user_inputs = {}
    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(important_features):
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())

        key_name = f"slider_{feature}"
        if key_name not in st.session_state:
            st.session_state[key_name] = mean_val

        if i % 3 == 0:
            user_inputs[feature] = col1.slider(
                feature,
                min_val,
                max_val,
                float(st.session_state[key_name]),
                key=key_name,
                step=float((max_val - min_val) / 100)
            )
        elif i % 3 == 1:
            user_inputs[feature] = col2.slider(
                feature,
                min_val,
                max_val,
                float(st.session_state[key_name]),
                key=key_name,
                step=float((max_val - min_val) / 100)
            )
        else:
            user_inputs[feature] = col3.slider(
                feature,
                min_val,
                max_val,
                float(st.session_state[key_name]),
                key=key_name,
                step=float((max_val - min_val) / 100)
            )

    if st.button("Rastgele Örnek Yükle"):
        st.session_state["load_random"] = True
        st.rerun()

    if st.button("Tahmin Yap"):
        # 1) Kullanıcının girdiği değerlerden DataFrame oluşturulmuş:
        input_df = pd.DataFrame([user_inputs])  # sütunlar = important_features

        # 2) Eğitimde oluşturulup kaydettiğiniz scaler'ı al
        scaler = st.session_state['scaler']

        # 3) Aynı scaler'ı uygulayıp ölçekle
        input_scaled = scaler.transform(input_df)

        # 4) Seçilen modelle tahmin et
        model = st.session_state['trained_models'][selected_model]
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        # 5) Sonuçları göster
        if pred == 1:
            st.error("Tanı: Malignant (Kötü Huylu)")
            prediction_text = "Kötü Huylu"
        else:
            st.success("Tanı: Benign (İyi Huylu)")
            prediction_text = "İyi Huylu"

        st.write(f"İyi Huylu Olasılığı: {proba[0]:.4f}")
        st.write(f"Kötü Huylu Olasılığı: {proba[1]:.4f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1],
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

        # SHAP ile açıklama bölümü
        st.subheader('❓ Neden Bu Tahmin?')
        try:
            # Modelin türüne göre SHAP explainer seçimi
            model_type = type(model).__name__
            
            # Random Forest ve Neural Network için özel hata mesajı
            if model_type in ['RandomForestClassifier', 'MLPClassifier']:
                st.warning(f"""
                ⚠️ SHAP grafiği {model_type} için şu anda görüntülenemiyor.
                
                **Sebep:** SHAP kütüphanesi sürüm uyumsuzluğu nedeniyle bu model türü için SHAP değerleri hesaplanamıyor.
                
                **Çözüm önerileri:**
                - SHAP kütüphanesini güncelleyin: `pip install --upgrade shap`
                - Alternatif olarak Lojistik Regresyon veya SVM modellerini kullanabilirsiniz.
                """)
            else:
                # Diğer modeller için normal SHAP işlemi
                if model_type in ['GradientBoostingClassifier', 'XGBClassifier']:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, st.session_state['X_train'])

                # SHAP değerlerini hesapla
                shap_values = explainer.shap_values(input_scaled)

                st.write('Aşağıda, bu tahminin nedenlerini gösteren SHAP grafiği yer almaktadır:')
                
                # SHAP grafiği açıklaması
                st.info("""
                **SHAP Grafiği Açıklaması:**
                - Grafikteki her çubuk, bir özelliğin tahmine olan katkısını gösterir
                - Kırmızı çubuklar tahmini kötü huylu yönde etkileyen özellikleri
                - Mavi çubuklar tahmini iyi huylu yönde etkileyen özellikleri temsil eder
                - Çubukların uzunluğu, özelliğin etkisinin büyüklüğünü gösterir
                - Base value (başlangıç değeri), tüm özelliklerin ortalama etkisini temsil eder
                """)
                
                # SHAP değerlerinin ve expected_value'nun yapısına göre uygun şekilde waterfall plot çiz
                try:
                    # Eğer shap_values bir liste ise (ör: [class0, class1]), binary classification için class1'i al
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_val = shap_values[1][0]
                        expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    else:
                        # Tek array ise
                        shap_val = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                        expected_val = explainer.expected_value[0] if (hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray))) else explainer.expected_value

                    fig_shap = shap.plots._waterfall.waterfall_legacy(expected_val, shap_val, feature_names=st.session_state['features'], show=False)
                    st.pyplot(fig_shap)
                except Exception as e:
                    # Force plot fallback
                    try:
                        shap_html = shap.plots.force(explainer.expected_value, shap_values[0], feature_names=st.session_state['features'], matplotlib=False)
                        components.html(shap_html, height=300)
                    except Exception as e2:
                        st.info(f'SHAP açıklaması oluşturulamadı: {e2}')
        except Exception as e:
            st.info(f'SHAP açıklaması oluşturulamadı: {e}') 