import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import calculate_metrics
from src.visualization import plot_confusion_matrix, plot_feature_importance

# --- Model Değerlendirme ve Sonuçlar Sayfası ---
def run_ml_evaluation():
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Lütfen önce Veri Ön İşleme adımını tamamlayın.")
        return
    st.header("Model Değerlendirme ve Sonuçlar")
    if 'trained_models' not in st.session_state:
        st.warning("Lütfen önce model eğitimini tamamlayın.")
        return
    # Tüm modellerin performanslarını karşılaştıran grafik
    st.subheader("Model Performans Karşılaştırması")
    all_metrics = []
    for model_name, model in st.session_state['trained_models'].items():
        y_pred = model.predict(st.session_state['X_test'])
        y_pred_proba = model.predict_proba(st.session_state['X_test'])[:, 1]
        metrics = calculate_metrics(
            st.session_state['y_test'], 
            y_pred,
            y_pred_proba
        )
        all_metrics.append({
            'Model': model_name,
            'Doğruluk': metrics['accuracy'],
            'Hassasiyet': metrics['precision'],
            'Duyarlılık': metrics['recall'],
            'F1 Skoru': metrics['f1'],
            'ROC AUC': metrics['roc_auc']
        })
    metrics_df = pd.DataFrame(all_metrics)
    selected_metric = st.selectbox(
        "Karşılaştırma için metrik seçin:",
        ['Doğruluk', 'Hassasiyet', 'Duyarlılık', 'F1 Skoru', 'ROC AUC'],
        index=0
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Model', y=selected_metric, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Model {selected_metric} Karşılaştırması')
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Tüm Metrikler")
    st.dataframe(metrics_df.set_index('Model').round(4))
    st.subheader("Detaylı Model Değerlendirmesi")
    model_names = list(st.session_state['trained_models'].keys())
    selected_model = st.selectbox("Değerlendirmek istediğiniz modeli seçin:", model_names)
    model = st.session_state['trained_models'][selected_model]
    y_pred = model.predict(st.session_state['X_test'])
    y_pred_proba = model.predict_proba(st.session_state['X_test'])[:, 1]
    metrics = calculate_metrics(
        st.session_state['y_test'], 
        y_pred,
        y_pred_proba
    )
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    roc_auc = metrics['roc_auc']
    confusion_mat = metrics['confusion_matrix']
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Doğruluk", f"{accuracy:.4f}")
    col2.metric("Hassasiyet", f"{precision:.4f}")
    col3.metric("Duyarlılık", f"{recall:.4f}")
    col4.metric("F1 Skoru", f"{f1:.4f}")
    col5.metric("ROC AUC", f"{roc_auc:.4f}")
    st.subheader("Karmaşıklık Matrisi")
    fig_cm = plot_confusion_matrix(confusion_mat)
    st.pyplot(fig_cm)
    st.subheader("ROC Eğrisi")
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        st.session_state['y_test'],
        y_pred_proba,
        name=selected_model,
        ax=ax
    )
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill')
    ax.set_title('ROC Eğrisi')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    st.subheader("Özellik Önemliliği")
    try:
        feature_importance_fig = plot_feature_importance(
            model, 
            st.session_state['features'],
            model_type=selected_model
        )
        st.pyplot(feature_importance_fig)
    except Exception as e:
        st.info(f"Bu model için özellik önemliliği görselleştirmesi mevcut değil. Hata: {e}")
    st.subheader("Yanlış Sınıflandırma Analizi")
    results_df = pd.DataFrame({
        'Gerçek Değer': st.session_state['y_test'],
        'Tahmin': y_pred,
        'Olasılık': y_pred_proba
    })
    misclassified = results_df[results_df['Gerçek Değer'] != results_df['Tahmin']]
    st.write(f"Yanlış sınıflandırılan örnek sayısı: {len(misclassified)}")
    if not misclassified.empty:
        st.dataframe(misclassified.sort_values(by='Olasılık', ascending=False), use_container_width=True) 