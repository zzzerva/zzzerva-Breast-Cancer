import streamlit as st
import pandas as pd
import numpy as np
from models.ml.ml_models import train_models, save_model
from src.data.data_processing import load_data
import matplotlib.pyplot as plt
import seaborn as sns

# --- Makine Öğrenmesi Algoritmaları Sayfası ---
def run_ml_algorithms():
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Lütfen önce Veri Ön İşleme adımını tamamlayın.")
        return
    st.header("Makine Öğrenmesi Algoritmaları")
    data = load_data("data.csv")
    st.info("Veri ön işleme ve özellik seçimi adımlarını tamamladıysanız, aşağıdan model seçip eğitebilirsiniz.")
    # Özellikler ve hedef
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        features = st.session_state['features']
    else:
        st.warning("Lütfen önce veri ön işleme ve özellik seçimi adımlarını tamamlayın.")
        return
    # Model seçimi
    models_to_train = st.multiselect(
        "Eğitmek istediğiniz modelleri seçin:",
        ["Lojistik Regresyon", "Random Forest", "Support Vector Machine", "Gradient Boosting", "Neural Network", "XGBoost"],
        default=["Lojistik Regresyon", "Random Forest", "Support Vector Machine"]
    )
    # Hiperparametre ayarları
    st.subheader("Hiperparametre Ayarları")
    hyper_params = {}
    if "Lojistik Regresyon" in models_to_train:
        st.write("#### Lojistik Regresyon Hiperparametreleri")
        c_value = st.slider("C (Düzenlileştirme)", 0.01, 10.0, 1.0, 0.01)
        hyper_params["Lojistik Regresyon"] = {"C": c_value}
    if "Random Forest" in models_to_train:
        st.write("#### Random Forest Hiperparametreleri")
        n_estimators = st.slider("Ağaç Sayısı", 10, 200, 100, 10)
        max_depth = st.slider("Maksimum Derinlik", 3, 20, 10, 1)
        hyper_params["Random Forest"] = {"n_estimators": n_estimators, "max_depth": max_depth}
    if "Support Vector Machine" in models_to_train:
        st.write("#### Support Vector Machine Hiperparametreleri")
        kernel = st.selectbox("Çekirdek Fonksiyonu", ["linear", "rbf", "poly", "sigmoid"], index=0)
        c_value_svm = st.slider("C (SVM)", 0.01, 10.0, 1.0, 0.01)
        hyper_params["Support Vector Machine"] = {"kernel": kernel, "C": c_value_svm}
    if "Gradient Boosting" in models_to_train:
        st.write("#### Gradient Boosting Hiperparametreleri")
        n_estimators_gb = st.slider("Ağaç Sayısı (GB)", 10, 200, 100, 10)
        learning_rate = st.slider("Öğrenme Oranı", 0.01, 0.5, 0.1, 0.01)
        hyper_params["Gradient Boosting"] = {"n_estimators": n_estimators_gb, "learning_rate": learning_rate}
    if "XGBoost" in models_to_train:
        st.write("#### XGBoost Hiperparametreleri")
        n_estimators_xgb = st.slider("Ağaç Sayısı (XGB)", 10, 200, 100, 10)
        learning_rate_xgb = st.slider("Öğrenme Oranı (XGB)", 0.01, 0.5, 0.1, 0.01)
        hyper_params["XGBoost"] = {"n_estimators": n_estimators_xgb, "learning_rate": learning_rate_xgb}
    if "Neural Network" in models_to_train:
        st.write("#### Yapay Sinir Ağı Hiperparametreleri")
        hidden_layer_sizes = st.text_input("Gizli Katman Boyutları (virgülle ayrılmış, örn: 64,32)", "32,16")
        activation = st.selectbox("Aktivasyon Fonksiyonu", ["relu", "tanh", "sigmoid"], index=0)
        hyper_params["Neural Network"] = {
            "hidden_layer_sizes": tuple(map(int, hidden_layer_sizes.split(','))), 
            "activation": activation
        }
    # Cross-validation seçimi
    cv_fold = st.slider("Çapraz Doğrulama Katlama Sayısı", 2, 10, 5, 1)
    # Model eğitimi
    if st.button("Modelleri Eğit"):
        if not models_to_train:
            st.error("Lütfen en az bir model seçin.")
        else:
            with st.spinner('Modeller eğitiliyor...'):
                trained_models, cv_results = train_models(
                    X_train,
                    y_train,
                    models_to_train,
                    hyper_params,
                    cv_fold
                )
                st.session_state['trained_models'] = trained_models
                st.session_state['cv_results'] = cv_results
                
                # Modelleri kaydet
                for model_name, model in trained_models.items():
                    file_path = f'models/saved_models/{model_name.lower().replace(" ", "_")}_model.pkl'
                    save_model(model, model_name, file_path)
                
                st.success("Modeller başarıyla eğitildi ve kaydedildi!")
                st.subheader("Çapraz Doğrulama Sonuçları:")
                st.dataframe(pd.DataFrame(cv_results), use_container_width=True)
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_df = pd.DataFrame(cv_results).sort_values(by='accuracy', ascending=False)
                sns.barplot(data=plot_df, x='model', y='accuracy', ax=ax)
                plt.xticks(rotation=30)
                plt.title('Modellere Göre Doğruluk')
                plt.tight_layout()
                st.pyplot(fig)
                # En iyi model
                best_row = plot_df.iloc[0]
                st.markdown(f"## En İyi Model: <span style='color:#43d17a'>{best_row['model']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Doğruluk:** <span style='color:#43d17a'>{best_row['accuracy']:.4f}</span>", unsafe_allow_html=True) 