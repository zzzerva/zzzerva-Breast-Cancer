import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.data.data_processing import load_data

def run_preprocessing():
    st.title("Veri Ön İşleme")
    data = load_data("data.csv")
    st.subheader("Veri Ön İşleme Adımları")
    # Eksik değerler
    st.write("### 1. Eksik Değer Analizi")
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.write("Eksik değerler bulunmaktadır:")
        st.write(missing_values[missing_values > 0])
    else:
        st.write("Veri setinde eksik değer bulunmamaktadır.")
    # Aykırı değerler
    st.write("### 2. Aykırı Değer Analizi")
    important_features = [
        'radius_mean', 'perimeter_mean', 'area_mean', 
        'concavity_mean', 'concave points_mean',
        'radius_worst', 'perimeter_worst', 'area_worst',
        'concavity_worst', 'concave points_worst'
    ]
    feature_for_outliers = st.selectbox(
        "Aykırı değer analizi için özellik seçin:",
        options=important_features,
        index=0
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='diagnosis', y=feature_for_outliers, ax=ax)
    ax.set_title(f'{feature_for_outliers} Özelliği için Aykırı Değerler')
    st.pyplot(fig)
    # Normalleştirme/Standartlaştırma
    st.write("### 3. Özellik Ölçeklendirme")
    st.write("""
    Veri setindeki özellikler farklı ölçeklere sahiptir. 
    Bu nedenle, modelin performansını artırmak için özellikler standartlaştırılmalı veya normalleştirilmelidir.
    Kullanılabilecek ölçeklendirme yöntemleri:
    - **Min-Max Normalizasyonu**: Özellikleri [0, 1] aralığına ölçeklendirir.
    - **Standartlaştırma (Z-Score)**: Özellikleri ortalama 0, standart sapma 1 olacak şekilde ölçeklendirir.
    """)
    scaling_method = st.radio(
        "Ölçeklendirme yöntemi seçin:",
        ["Min-Max Normalizasyonu", "Standartlaştırma (Z-Score)"]
    )
    if st.button("Veriyi Ön İşle"):
        with st.spinner('Veri ön işleniyor...'):
            if scaling_method == "Min-Max Normalizasyonu":
                scaling_method_param = "minmax"
            else:
                scaling_method_param = "standardization"
            important_features = [
                'radius_mean', 'perimeter_mean', 'area_mean', 
                'concavity_mean', 'concave points_mean',
                'radius_worst', 'perimeter_worst', 'area_worst',
                'concavity_worst', 'concave points_worst'
            ]
            X = data[important_features].values
            y = (data['diagnosis'] == 'M').astype(int)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if scaling_method_param == "standardization":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.success("Veri başarıyla ön işlendi!")
            st.session_state['preprocessing_done'] = True
            st.subheader("İşlenmiş Veri")
            processed_data = pd.DataFrame(X_train, columns=important_features)
            processed_data['diagnosis'] = y_train
            st.dataframe(processed_data.head())
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['features'] = important_features
            st.session_state['scaling_method'] = scaling_method_param
            st.session_state['scaler'] = scaler
            st.write("Eğitim Veri Boyutu:", X_train.shape)
            st.write("Test Veri Boyutu:", X_test.shape)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.countplot(x=y_train, ax=ax[0])
            ax[0].set_title('Eğitim Seti Sınıf Dağılımı')
            ax[0].set_xlabel('Tanı (0: İyi Huylu, 1: Kötü Huylu)')
            ax[0].set_ylabel('Sayı')
            sns.countplot(x=y_test, ax=ax[1])
            ax[1].set_title('Test Seti Sınıf Dağılımı')
            ax[1].set_xlabel('Tanı (0: İyi Huylu, 1: Kötü Huylu)')
            ax[1].set_ylabel('Sayı')
            plt.tight_layout()
            st.pyplot(fig)