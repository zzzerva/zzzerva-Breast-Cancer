import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.data.data_processing import load_data, clean_data
from src.visualization import plot_correlation_matrix

def run_data_analysis():
    st.title("Meme Kanseri Veri Seti Analizi")
    st.write("Bu sayfa, Wisconsin Meme Kanseri veri setinin kapsamlı bir analizini sunar.")

    data = load_data("data.csv")
    cleaned_data = clean_data(data)

    st.header("Veri Seti Genel Bilgiler")
    tab1, tab2, tab3 = st.tabs(["Veri Özeti", "İstatistikler", "Görselleştirmeler"])

    with tab1:
        st.subheader("Veri Seti Boyutu")
        st.write(f"Satır sayısı: {cleaned_data.shape[0]}, Sütun sayısı: {cleaned_data.shape[1]}")
        st.subheader("İlk Birkaç Satır")
        st.dataframe(cleaned_data.head())
        st.subheader("Sütun Bilgileri")
        cols = pd.DataFrame({
            'Sütun': cleaned_data.columns,
            'Veri Tipi': cleaned_data.dtypes,
            'Boş Değerler': cleaned_data.isnull().sum(),
            'Benzersiz Değerler': [cleaned_data[col].nunique() for col in cleaned_data.columns]
        })
        st.dataframe(cols)
        st.subheader("Hedef Değişken Dağılımı")
        target_counts = cleaned_data['diagnosis'].value_counts().reset_index()
        target_counts.columns = ['Tanı', 'Sayı']
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(target_counts)
        with col2:
            fig = px.pie(
                target_counts, values='Sayı', names='Tanı',
                title="Tanı Dağılımı (0: Benign, 1: Malignant)",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Tanımlayıcı İstatistikler")
        st.dataframe(cleaned_data.describe())
        feature_groups = {
            'Mean Özellikleri': [col for col in cleaned_data.columns if 'mean' in col],
            'SE Özellikleri': [col for col in cleaned_data.columns if '_se' in col],
            'Worst Özellikleri': [col for col in cleaned_data.columns if 'worst' in col]
        }
        selected_group = st.selectbox("Özellik Grubu Seçin:", list(feature_groups.keys()), key="feature_group_select")
        if selected_group:
            group_cols = feature_groups[selected_group]
            st.dataframe(cleaned_data[group_cols].describe())

    with tab3:
        st.subheader("Görselleştirmeler")
        viz_type = st.selectbox(
            "Görselleştirme Türü:",
            ["Korelasyon Matrisi", "Kutu Grafikleri", "Dağılım Grafikleri", "3D Dağılım"],
            key="viz_type_select"
        )
        if viz_type == "Korelasyon Matrisi":
            st.write("Özellikler arasındaki korelasyon:")
            fig = plot_correlation_matrix(cleaned_data, figsize=(14, 12), mask_upper=True, annot=False)
            st.pyplot(fig)
            st.subheader("Hedef Değişkenle En Yüksek Korelasyona Sahip 10 Özellik")
            correlation_matrix = cleaned_data.corr()
            target_corr = correlation_matrix['diagnosis'].drop('diagnosis').sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x=target_corr.index[:10], y=target_corr.values[:10], palette='coolwarm')
            plt.xticks(rotation=90)
            plt.title('Tanı ile En Yüksek Korelasyona Sahip 10 Özellik')
            plt.tight_layout()
            st.pyplot(fig)
        elif viz_type == "Kutu Grafikleri":
            st.write("Tanı gruplarına göre özellik dağılımları:")
            all_features = cleaned_data.drop('diagnosis', axis=1).columns.tolist()
            selected_features = st.multiselect(
                "Görselleştirilecek özellikleri seçin (en fazla 5):",
                all_features,
                default=all_features[:3],
                key="selected_features_select"
            )
            if len(selected_features) > 5:
                st.warning("En fazla 5 özellik seçebilirsiniz. İlk 5 özellik gösterilecek.")
                selected_features = selected_features[:5]
            if selected_features:
                fig, axes = plt.subplots(
                    nrows=len(selected_features), figsize=(12, 4 * len(selected_features))
                )
                if len(selected_features) == 1:
                    axes = [axes]
                for i, feature in enumerate(selected_features):
                    sns.boxplot(x='diagnosis', y=feature, data=cleaned_data, palette='Set2', ax=axes[i])
                    axes[i].set_title(f'{feature} - Tanı İlişkisi')
                    axes[i].set_xlabel('Tanı (0: Benign, 1: Malignant)')
                plt.tight_layout()
                st.pyplot(fig)
        elif viz_type == "Dağılım Grafikleri":
            st.write("Özellik dağılımları:")
            all_features = cleaned_data.drop('diagnosis', axis=1).columns.tolist()
            x_feature = st.selectbox("X ekseni için özellik seçin:", all_features, index=0, key="x_feature_select")
            y_feature = st.selectbox("Y ekseni için özellik seçin:", all_features, index=1, key="y_feature_select")
            if x_feature and y_feature:
                fig = px.scatter(
                    cleaned_data, x=x_feature, y=y_feature, color='diagnosis',
                    color_discrete_sequence=['#3498db', '#e74c3c'],
                    title=f"{x_feature} vs {y_feature}",
                    labels={'diagnosis': 'Tanı (0: Benign, 1: Malignant)'}
                )
                st.plotly_chart(fig)
        elif viz_type == "3D Dağılım":
            st.write("3 boyutlu özellik dağılımı:")
            all_features = cleaned_data.drop('diagnosis', axis=1).columns.tolist()
            features_3d = st.multiselect(
                "3D için 3 özellik seçin:",
                all_features,
                default=all_features[:3],
                key="features_3d_select"
            )
            if len(features_3d) == 3:
                fig = px.scatter_3d(
                    cleaned_data, x=features_3d[0], y=features_3d[1], z=features_3d[2],
                    color='diagnosis',
                    color_discrete_sequence=['#3498db', '#e74c3c'],
                    title=f"3D: {features_3d[0]} vs {features_3d[1]} vs {features_3d[2]}",
                    labels={'diagnosis': 'Tanı (0: Benign, 1: Malignant)'}
                )
                st.plotly_chart(fig) 