import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap
import streamlit.components.v1 as components

# Modülleri ekle
sys.path.append("src")
sys.path.append("utils")
sys.path.append("models")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# İçe aktarmalar
from src.data.data_processing import preprocess_data, clean_data
from src.visualization import plot_correlation_matrix, plot_feature_importance, plot_confusion_matrix, plot_3d_scatter
from models.ml.ml_models import train_models, evaluate_models
from utils.metrics import calculate_metrics
from src.data.data_processing import load_data
from applicate.modules.data_analysis import run_data_analysis
from applicate.modules.data_preprocessing import run_preprocessing
from applicate.modules.feature_engineering import run_feature_engineering
from applicate.modules.ml_algorithms import run_ml_algorithms
from applicate.modules.ml_evaluation import run_ml_evaluation
from applicate.modules.ml_predict import run_ml_predict
from applicate.modules.dl_algorithms import run_dl_algorithms
from applicate.modules.dl_evaluation import run_dl_evaluation
from applicate.modules.dl_predict import run_dl_predict

# Sayfa yapılandırması
st.set_page_config(
    page_title="Meme Kanseri Tahmini",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar'ın en üstünde ana adımlar
ana_adim = st.sidebar.radio('Adımlar', [
    'Veri Analizi',
    'Veri Ön İşleme',
    'Makine Öğrenmesi',
    'Derin Öğrenme'
])

if ana_adim == 'Veri Analizi':
    run_data_analysis()

elif ana_adim == 'Veri Ön İşleme':
    run_preprocessing()

elif ana_adim == 'Makine Öğrenmesi':
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Makine Öğrenmesi Adımı**')
    ml_adim = st.sidebar.radio('', [
        'Özellik Seçimi ve Mühendisliği',
        'ML Algoritmaları',
        'ML Model Değerlendirme ve Sonuçlar',
        'ML Tahmin'
    ])
    if ml_adim == 'Özellik Seçimi ve Mühendisliği':
        run_feature_engineering()
    elif ml_adim == 'ML Algoritmaları':
        run_ml_algorithms()
    elif ml_adim == 'ML Model Değerlendirme ve Sonuçlar':
        run_ml_evaluation()
    elif ml_adim == 'ML Tahmin':
        run_ml_predict()

elif ana_adim == 'Derin Öğrenme':
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Lütfen önce 'Veri Ön İşleme' adımını tamamlayın. Veri ön işleme yapılmadan derin öğrenme adımlarına geçemezsiniz.")
    else:
        st.sidebar.markdown('---')
        st.sidebar.markdown('**Derin Öğrenme Adımı**')
        dl_adim = st.sidebar.radio('', [
            'DL Algoritmaları',
            'DL Model Değerlendirme ve Sonuçlar',
            'DL Tahmin'
        ])
        if dl_adim == 'DL Algoritmaları':
            run_dl_algorithms()
        elif dl_adim == 'DL Model Değerlendirme ve Sonuçlar':
            run_dl_evaluation()
        elif dl_adim == 'DL Tahmin':
            run_dl_predict()
