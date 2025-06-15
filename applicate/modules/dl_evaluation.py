import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.data.data_processing import load_data

def run_dl_evaluation():
    st.title("Derin Öğrenme Model Değerlendirme")
    
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
    
    # Veri ön işleme
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model değerlendirme
    st.subheader("Model Performans Metrikleri")
    
    # Tahminler
    X_test_scaled = scaler.transform(X_test)
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrikler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'Doğruluk': accuracy_score(y_test, y_pred),
        'Kesinlik': precision_score(y_test, y_pred),
        'Duyarlılık': recall_score(y_test, y_pred),
        'F1 Skoru': f1_score(y_test, y_pred)
    }
    
    # Metrikleri göster
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Doğruluk", f"{metrics['Doğruluk']:.4f}")
        st.metric("Kesinlik", f"{metrics['Kesinlik']:.4f}")
    with col2:
        st.metric("Duyarlılık", f"{metrics['Duyarlılık']:.4f}")
        st.metric("F1 Skoru", f"{metrics['F1 Skoru']:.4f}")
    
    # Sınıflandırma raporu
    st.subheader("Sınıflandırma Raporu")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Görselleştirmeler
    st.subheader("Model Değerlendirme Grafikleri")
    
    # Confusion Matrix
    st.write("Karmaşıklık Matrisi")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    st.pyplot(plt)
    
    # ROC Eğrisi
    st.write("ROC Eğrisi")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    
    # Eğitim grafiği (MLP için)
    if 'history' in model_info:
        st.subheader("Eğitim Geçmişi")
        history = model_info['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss grafiği
        ax1.plot(history.history['loss'], label='Eğitim Kaybı')
        ax1.plot(history.history['val_loss'], label='Doğrulama Kaybı')
        ax1.set_title('Model Kaybı')
        ax1.set_xlabel('Devir')
        ax1.set_ylabel('Kayıp')
        ax1.legend()
        
        # Accuracy grafiği
        ax2.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
        ax2.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        ax2.set_title('Model Doğruluğu')
        ax2.set_xlabel('Devir')
        ax2.set_ylabel('Doğruluk')
        ax2.legend()
        
        st.pyplot(fig)
    
    # Sonraki adım için yönlendirme
    st.info("Model değerlendirmesi tamamlandı. Tahmin yapmak için 'DL Tahmin' sayfasına geçebilirsiniz.") 