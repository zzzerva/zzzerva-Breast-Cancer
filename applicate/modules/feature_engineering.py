import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from src.data.data_processing import load_data, clean_data

# --- Özellik Seçimi ve Mühendisliği Sayfası ---
def run_feature_engineering():
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Lütfen önce Veri Ön İşleme adımını tamamlayın.")
        return
    st.header("Özellik Seçimi ve Mühendisliği")
    data = load_data("data.csv")
    cleaned_data = clean_data(data)
    st.subheader("Korelasyon Analizi")
    corr = cleaned_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Korelasyon Matrisi', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("SelectKBest ile Özellik Seçimi")
    X = cleaned_data.drop('diagnosis', axis=1)
    y = cleaned_data['diagnosis']
    k = st.slider("Seçilecek Özellik Sayısı (K)", min_value=2, max_value=min(20, X.shape[1]), value=10)
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    st.write(f"Seçilen Özellikler: {', '.join(selected_features)}")
    st.dataframe(cleaned_data[selected_features].head())
    st.subheader("PCA ile Boyut İndirgeme")
    n_components = st.slider("PCA Bileşen Sayısı", min_value=2, max_value=min(10, X.shape[1]), value=2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_)
    st.write(f"Açıklanan Toplam Varyans: {explained_var:.2%}")
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
        ax.set_xlabel('Birinci Bileşen')
        ax.set_ylabel('İkinci Bileşen')
        ax.set_title('PCA - 2 Bileşen')
        legend1 = ax.legend(*scatter.legend_elements(), title="Tanı")
        ax.add_artist(legend1)
        st.pyplot(fig)
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='coolwarm', alpha=0.7)
        ax.set_xlabel('Birinci Bileşen')
        ax.set_ylabel('İkinci Bileşen')
        ax.set_zlabel('Üçüncü Bileşen')
        ax.set_title('PCA - 3 Bileşen')
        fig.colorbar(scatter, ax=ax, label='Tanı')
        st.pyplot(fig)
    st.info("Seçtiğiniz özellikler ve boyut indirgeme sonuçları, model eğitiminde kullanılmak üzere session_state'e kaydedilebilir.") 