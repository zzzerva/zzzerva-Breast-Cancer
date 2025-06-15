import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_models(X_train, y_train, models_to_train, hyper_params=None, cv_fold=5, random_state=42):
    """
    Modelleri eğitir
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Eğitim verileri
    y_train : numpy.ndarray
        Eğitim etiketleri
    models_to_train : list
        Eğitilecek modellerin listesi
    hyper_params : dict, optional
        Model hiperparametreleri
    cv_fold : int, default=5
        Çapraz doğrulama katlama sayısı
    
    Returns:
    --------
    trained_models : dict
        Eğitilmiş modeller
    cv_results : list
        Çapraz doğrulama sonuçları
    """
    if hyper_params is None:
        hyper_params = {}
    
    trained_models = {}
    cv_results = []
    
    for model_name in models_to_train:
        if model_name == "Lojistik Regresyon":
            model = LogisticRegression(
                C=hyper_params.get(model_name, {}).get('C', 1.0),
                random_state=random_state
            )
        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=hyper_params.get(model_name, {}).get('n_estimators', 100),
                max_depth=hyper_params.get(model_name, {}).get('max_depth', None),
                random_state=random_state
            )
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=hyper_params.get(model_name, {}).get('max_depth', None),
                random_state=random_state
            )
        elif model_name == "Support Vector Machine":
            model = SVC(
                kernel=hyper_params.get(model_name, {}).get('kernel', 'rbf'),
                C=hyper_params.get(model_name, {}).get('C', 1.0),
                probability=True,
                random_state=random_state
            )
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=hyper_params.get(model_name, {}).get('n_estimators', 100),
                learning_rate=hyper_params.get(model_name, {}).get('learning_rate', 0.1),
                random_state=random_state
            )
        elif model_name == "XGBoost":
            model = XGBClassifier(
                n_estimators=hyper_params.get(model_name, {}).get('n_estimators', 100),
                learning_rate=hyper_params.get(model_name, {}).get('learning_rate', 0.1),
                random_state=random_state
            )
        elif model_name == "Neural Network":
            model = MLPClassifier(
                hidden_layer_sizes=hyper_params.get(model_name, {}).get('hidden_layer_sizes', (100,)),
                activation=hyper_params.get(model_name, {}).get('activation', 'relu'),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state
            )
        else:
            continue
        
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)

        # Modeli eğit
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Çapraz doğrulama
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_fold, scoring='accuracy')
        cv_results.append({
            'model': model_name,
            'accuracy': cv_scores.mean(),
            'std': cv_scores.std()
        })
    
    return trained_models, cv_results

def evaluate_models(models, X_test, y_test):
    """
    Modelleri değerlendirir
    
    Parameters:
    -----------
    models : dict
        Eğitilmiş modellerin sözlüğü
    X_test : numpy.ndarray
        Test veri seti
    y_test : numpy.ndarray
        Test etiketleri
        
    Returns:
    --------
    model_metrics : dict
        Model metriklerinin sözlüğü
    """
    model_metrics = {}
    
    for model_name, model in models.items():
        print(f"{model_name} değerlendiriliyor...")
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC ve AUC
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
        else:
            fpr, tpr, auc_score = None, None, None
            
        model_metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score,
            "y_pred": y_pred
        }
        
        print(f"{model_name} değerlendirmesi tamamlandı.")
        print(f"Doğruluk: {accuracy:.4f}, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}, F1: {f1:.4f}")
        
    return model_metrics

def get_cross_validation_scores(models, X, y, cv=5, scoring='accuracy'):
    """
    Çapraz doğrulama skorlarını hesaplar
    
    Parameters:
    -----------
    models : dict
        Eğitilmiş modellerin sözlüğü
    X : numpy.ndarray
        Veri seti
    y : numpy.ndarray
        Etiketler
    cv : int, default=5
        Çapraz doğrulama sayısı
    scoring : str, default='accuracy'
        Skorlama metriği
        
    Returns:
    --------
    cv_scores : dict
        Çapraz doğrulama skorlarının sözlüğü
    """
    cv_scores = {}
    
    for model_name, model in models.items():
        print(f"{model_name} için çapraz doğrulama yapılıyor...")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_scores[model_name] = {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores)
        }
        print(f"{model_name} çapraz doğrulama sonucu: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    return cv_scores

def get_learning_curves(models, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Öğrenme eğrilerini hesaplar
    
    Parameters:
    -----------
    models : dict
        Eğitilmiş modellerin sözlüğü
    X : numpy.ndarray
        Veri seti
    y : numpy.ndarray
        Etiketler
    cv : int, default=5
        Çapraz doğrulama sayısı
    train_sizes : numpy.ndarray, default=np.linspace(0.1, 1.0, 10)
        Eğitim seti boyutları
        
    Returns:
    --------
    learning_curves : dict
        Öğrenme eğrilerinin sözlüğü
    """
    learning_curves_data = {}
    
    for model_name, model in models.items():
        print(f"{model_name} için öğrenme eğrisi hesaplanıyor...")
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, scoring='accuracy'
        )
        
        learning_curves_data[model_name] = {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "test_scores": test_scores
        }
        
    return learning_curves_data

def get_best_model(model_metrics, metric='f1'):
    """
    En iyi modeli seçer
    
    Parameters:
    -----------
    model_metrics : dict
        Model metriklerinin sözlüğü
    metric : str, default='f1'
        Karşılaştırma metriği
        
    Returns:
    --------
    best_model_name : str
        En iyi model adı
    best_score : float
        En iyi skor
    """
    best_score = -1
    best_model_name = None
    
    for model_name, metrics in model_metrics.items():
        if metrics[metric] > best_score:
            best_score = metrics[metric]
            best_model_name = model_name
    
    return best_model_name, best_score

def save_model(model, model_name, file_path):
    """
    Modeli kaydeder
    
    Parameters:
    -----------
    model : sklearn model
        Kaydedilecek model
    model_name : str
        Model adı
    file_path : str
        Kayıt yolu
        
    Returns:
    --------
    success : bool
        İşlem başarısı
    """
    import pickle
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{model_name} modeli başarıyla kaydedildi: {file_path}")
        return True
    except Exception as e:
        print(f"Model kaydedilirken hata oluştu: {e}")
        return False

def load_model(file_path):
    """
    Modeli yükler
    
    Parameters:
    -----------
    file_path : str
        Yükleme yolu
        
    Returns:
    --------
    model : sklearn model
        Yüklenen model
    """
    import pickle
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model başarıyla yüklendi: {file_path}")
        return model
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None

def build_pipeline(preprocessor, model):
    """
    Ön işleme ve model için boru hattı oluşturur
    
    Parameters:
    -----------
    preprocessor : sklearn.preprocessing
        Önişleme adımı
    model : sklearn model
        Model
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Oluşturulan boru hattı
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def load_ml_model(model_type):
    """Belirtilen ML modelini yükle"""
    model_path = f'models/saved_models/{model_type}.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def train_ml_model(model_type, X_train, y_train):
    """Belirtilen ML modelini eğit"""
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Modeli eğit
    model.fit(X_train_scaled, y_train)
    
    # Modeli ve scaler'ı kaydet
    os.makedirs('models/saved_models', exist_ok=True)
    joblib.dump((model, scaler), f'models/saved_models/{model_type}.joblib')
    
    return model

def predict_ml(model_type, X):
    """ML modeli ile tahmin yap"""
    model_data = load_ml_model(model_type)
    if model_data is None:
        return None
    
    model, scaler = model_data
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled) 