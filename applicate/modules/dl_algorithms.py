import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_processing import load_data
from models.dl.deep_learning import create_mlp_model, create_tabnet_model, train_dl_model, save_dl_model
import time

def run_dl_algorithms():
    st.title("Derin Öğrenme Algoritmaları")
    
    # Veri yükleme
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
    
    # Veriyi session state'e kaydet
    st.session_state['X_train'] = X_train
    st.session_state['features'] = X.columns.tolist()
    
    # Model seçimi
    model_type = st.selectbox(
        "Model Tipi",
        ["MLP", "TabNet"],
        help="Kullanmak istediğiniz derin öğrenme modelini seçin"
    )
    
    # Model parametreleri
    st.subheader("Model Parametreleri")
    
    if model_type == "MLP":
        epochs = st.slider("Epochs", 10, 100, 50)
        batch_size = st.slider("Batch Size", 8, 64, 32)
        hidden_layers = st.multiselect(
            "Gizli Katmanlar",
            [32, 64, 128, 256],
            default=[64, 32],
            help="Gizli katmanların nöron sayılarını seçin"
        )
        dropout_rate = st.slider("Dropout Oranı", 0.0, 0.5, 0.3, 0.1)
        
    elif model_type == "TabNet":
        epochs = st.slider("Epochs", 10, 100, 50)
        batch_size = st.slider("Batch Size", 8, 64, 32)
        n_d = st.slider("n_d", 4, 16, 8)
        n_a = st.slider("n_a", 4, 16, 8)
        n_steps = st.slider("n_steps", 1, 5, 3)
    
    # Model eğitimi
    if st.button("Modeli Eğit"):
        with st.spinner("Model eğitiliyor..."):
            start_time = time.time()
            
            # Model oluştur ve eğit
            model, history, scaler = train_dl_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size
            )
            
            training_time = time.time() - start_time
            
            # Modeli ve scaler'ı kaydet
            try:
                save_dl_model(model=model, model_type=model_type, scaler=scaler)
            except Exception as e:
                st.error(f"Model kaydedilirken hata oluştu: {str(e)}")
                return
            
            # Session state'e kaydet
            if 'trained_models' not in st.session_state:
                st.session_state['trained_models'] = {}
            st.session_state['trained_models']['Deep Learning'] = {
                'model': model,
                'model_type': model_type,
                'scaler': scaler
            }
            
            st.success(f"Model eğitimi tamamlandı! (Süre: {training_time:.2f} saniye)")
            
            # Eğitim grafiği (MLP için)
            if model_type == "MLP" and history is not None:
                st.subheader("Eğitim Grafiği")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Loss grafiği
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                # Accuracy grafiği
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Model Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                
                st.pyplot(fig)
            
            # Sonraki adım için yönlendirme
            st.info("Model eğitimi tamamlandı. Model performansını değerlendirmek için 'Model Değerlendirme' sayfasına geçebilirsiniz.") 