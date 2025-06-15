import os

# Veri yolları
DATA_PATH = "data.csv"
MODEL_SAVE_PATH = "models/saved_models"
FIGURE_SAVE_PATH = "static/figures"

# Veri ön işleme ayarları
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # Eğitim setinin %25'i doğrulama için kullanılacak
RANDOM_STATE = 42

# Model eğitim parametreleri
# Geleneksel modeller
TRADITIONAL_MODELS = [
    "Lojistik Regresyon",
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Decision Tree",
    "XGBoost",
    "Neural Network"
]

# XGBoost parametreleri
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0
}

# Random Forest parametreleri
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
}

# SVM parametreleri
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True
}

# Lojistik Regresyon parametreleri
LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000
}

# Derin öğrenme model parametreleri
DL_HIDDEN_LAYERS = [64, 32, 16]
DL_DROPOUT_RATE = 0.3
DL_ACTIVATION = "relu"
DL_LEARNING_RATE = 0.001
DL_BATCH_SIZE = 32
DL_EPOCHS = 100
DL_PATIENCE = 10

# Özellik mühendisliği ayarları
FEATURE_ENGINEERING = {
    "scaling": "StandardScaler",  # Alternatif: "MinMaxScaler"
    "feature_selection": "SelectKBest",
    "k_best_features": 15,
    "pca_components": 10
}

# Streamlit ayarları
STREAMLIT_TITLE = "Meme Kanseri Tahmini"
STREAMLIT_DESCRIPTION = """
Bu uygulama, Wisconsin Meme Kanseri veri seti üzerinde eğitilmiş makine öğrenmesi modelleri 
kullanarak meme kanseri tahmini yapabilmektedir. Hücre çekirdeği özelliklerine dayalı olarak 
benign (iyi huylu) veya malignant (kötü huylu) tümör tahmini yapabilirsiniz.
"""

# Görselleştirme ayarları
VISUALIZATION_CONFIG = {
    "figsize": (12, 8),
    "title_fontsize": 16,
    "label_fontsize": 12,
    "correlation_cmap": "coolwarm",
    "categorical_palette": "Set2",
    "continuous_palette": "viridis"
}

# Fonksiyonlar
def create_directories():
    """
    Gerekli dizinleri oluşturur
    """
    directories = [
        MODEL_SAVE_PATH,
        FIGURE_SAVE_PATH
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"{directory} dizini oluşturuldu veya zaten var.")

def get_model_path(model_name):
    """
    Model kayıt yolunu döndürür
    
    Parameters:
    -----------
    model_name : str
        Model adı
        
    Returns:
    --------
    str
        Model kayıt yolu
    """
    # Boşlukları alt çizgi ile değiştir ve küçük harfe çevir
    filename = model_name.lower().replace(" ", "_")
    return os.path.join(MODEL_SAVE_PATH, f"{filename}.pkl") 