import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Sınıflandırma metriklerini hesaplar
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred : numpy.ndarray
        Tahmin edilen değerler
    y_prob : numpy.ndarray, default=None
        Tahmin olasılıkları
        
    Returns:
    --------
    metrics : dict
        Metriklerin sözlüğü
    """
    metrics = {}
    
    # Temel metrikler
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Karmaşıklık matrisi
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Olasılık tabanlı metrikler
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Precision-Recall metriği
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
        
        # ROC ve Precision-Recall eğrileri için veriler
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
    
    return metrics

def display_classification_report(y_true, y_pred):
    """
    Sınıflandırma raporunu gösterir
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred : numpy.ndarray
        Tahmin edilen değerler
        
    Returns:
    --------
    report : str
        Sınıflandırma raporu
    """
    return classification_report(y_true, y_pred)

def calculate_fpr_tpr_thresholds(y_true, y_prob):
    """
    ROC eğrisi için FPR, TPR ve eşik değerlerini hesaplar
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_prob : numpy.ndarray
        Tahmin olasılıkları
        
    Returns:
    --------
    fpr : numpy.ndarray
        False positive rate
    tpr : numpy.ndarray
        True positive rate
    thresholds : numpy.ndarray
        Eşik değerleri
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds

def find_best_threshold(fpr, tpr, thresholds):
    """
    Youden indeksine göre en iyi eşik değerini bulur
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rate
    tpr : numpy.ndarray
        True positive rate
    thresholds : numpy.ndarray
        Eşik değerleri
        
    Returns:
    --------
    best_threshold : float
        En iyi eşik değeri
    """
    # Youden indeksini hesapla (J = sensitivity + specificity - 1)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]

def compare_models(model_metrics):
    """
    Farklı modelleri karşılaştırır
    
    Parameters:
    -----------
    model_metrics : dict
        Her model için metriklerin sözlüğü
        
    Returns:
    --------
    comparison_df : pandas.DataFrame
        Karşılaştırma tablosu
    """
    models = list(model_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    data = []
    for model_name in models:
        model_data = [model_name]
        for metric in metrics:
            if metric in model_metrics[model_name]:
                model_data.append(model_metrics[model_name][metric])
            else:
                model_data.append(np.nan)
        data.append(model_data)
    
    comparison_df = pd.DataFrame(data, columns=['Model'] + metrics)
    
    return comparison_df

def get_best_model_by_metric(model_metrics, metric='f1'):
    """
    Belirli bir metriğe göre en iyi modeli bulur
    
    Parameters:
    -----------
    model_metrics : dict
        Her model için metriklerin sözlüğü
    metric : str, default='f1'
        Karşılaştırma metriği
        
    Returns:
    --------
    best_model : str
        En iyi model adı
    best_value : float
        En iyi metrik değeri
    """
    best_value = -np.inf
    best_model = None
    
    for model_name, metrics in model_metrics.items():
        if metric in metrics and metrics[metric] > best_value:
            best_value = metrics[metric]
            best_model = model_name
    
    return best_model, best_value

def calculate_mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    İki model arasında McNemar testi uygular
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred_model1 : numpy.ndarray
        İlk modelin tahminleri
    y_pred_model2 : numpy.ndarray
        İkinci modelin tahminleri
        
    Returns:
    --------
    statistic : float
        İstatistik değeri
    p_value : float
        p değeri
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Tahminlerin doğruluğunu hesapla
    correct_model1 = (y_true == y_pred_model1)
    correct_model2 = (y_true == y_pred_model2)
    
    # Koşullu tablo oluştur
    table = np.zeros((2, 2), dtype=int)
    table[0, 0] = np.sum((correct_model1) & (correct_model2))  # Her iki model de doğru
    table[0, 1] = np.sum((correct_model1) & (~correct_model2))  # Model 1 doğru, Model 2 yanlış
    table[1, 0] = np.sum((~correct_model1) & (correct_model2))  # Model 1 yanlış, Model 2 doğru
    table[1, 1] = np.sum((~correct_model1) & (~correct_model2))  # Her iki model de yanlış
    
    # McNemar testi uygula
    result = mcnemar(table, exact=False, correction=True)
    
    return result.statistic, result.pvalue

def calculate_bootstrap_confidence_interval(y_true, y_pred, n_iterations=1000, alpha=0.05):
    """
    Bootstrap yöntemiyle güven aralığı hesaplar
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred : numpy.ndarray
        Tahmin edilen değerler
    n_iterations : int, default=1000
        Bootstrap örnekleme sayısı
    alpha : float, default=0.05
        Anlamlılık düzeyi
        
    Returns:
    --------
    ci : dict
        Her metrik için güven aralıkları
    """
    metrics = {}
    
    # Her metrik için bootstrap skorlarını tut
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    
    indices = np.arange(len(y_true))
    for _ in range(n_iterations):
        # Bootstrap örneklem
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_y_true = y_true[bootstrap_indices]
        bootstrap_y_pred = y_pred[bootstrap_indices]
        
        # Metrikleri hesapla
        acc_scores.append(accuracy_score(bootstrap_y_true, bootstrap_y_pred))
        prec_scores.append(precision_score(bootstrap_y_true, bootstrap_y_pred))
        rec_scores.append(recall_score(bootstrap_y_true, bootstrap_y_pred))
        f1_scores.append(f1_score(bootstrap_y_true, bootstrap_y_pred))
    
    # Güven aralıklarını hesapla
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    metrics['accuracy'] = {
        'mean': np.mean(acc_scores),
        'ci_lower': np.percentile(acc_scores, lower_percentile),
        'ci_upper': np.percentile(acc_scores, upper_percentile)
    }
    
    metrics['precision'] = {
        'mean': np.mean(prec_scores),
        'ci_lower': np.percentile(prec_scores, lower_percentile),
        'ci_upper': np.percentile(prec_scores, upper_percentile)
    }
    
    metrics['recall'] = {
        'mean': np.mean(rec_scores),
        'ci_lower': np.percentile(rec_scores, lower_percentile),
        'ci_upper': np.percentile(rec_scores, upper_percentile)
    }
    
    metrics['f1'] = {
        'mean': np.mean(f1_scores),
        'ci_lower': np.percentile(f1_scores, lower_percentile),
        'ci_upper': np.percentile(f1_scores, upper_percentile)
    }
    
    return metrics 