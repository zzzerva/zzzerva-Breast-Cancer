import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import plotly.express as px

def plot_correlation_matrix(data, figsize=(12, 10), mask_upper=True, annot=False):
    """
    Korelasyon matrisi çizimi
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Korelasyon matrisi hesaplanacak veri
    figsize : tuple, default=(12, 10)
        Şekil boyutu
    mask_upper : bool, default=True
        Üst üçgeni maskele
    annot : bool, default=False
        Korelasyon değerlerini göster
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    # Korelasyon matrisini hesapla
    corr = data.corr()
    
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # Maske oluştur
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    else:
        mask = None
    
    # Isı haritası çiz
    sns.heatmap(corr, mask=mask, annot=annot, cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    plt.title('Korelasyon Matrisi', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, feature_names, model_type="", top_n=20, figsize=(10, 8)):
    """
    Özellik önemini görselleştirir
    
    Parameters:
    -----------
    model : sklearn model
        Eğitilmiş model
    feature_names : list
        Özellik isimleri
    model_type : str, default=""
        Model tipi
    top_n : int, default=20
        Gösterilecek özellik sayısı
    figsize : tuple, default=(10, 8)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    # Farklı model türleri için özellik önemini al
    if model_type in ["Random Forest", "XGBoost", "Gradient Boosting"]:
        importances = model.feature_importances_
    elif model_type == "Lojistik Regresyon":
        importances = np.abs(model.coef_[0])
    elif model_type == "Support Vector Machine" and hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Model tipi '{model_type}' için özellik önemi hesaplanamıyor.")
    
    # Özellik önemlerini sırala
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # Çubuk grafiği çiz
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Önem Derecesi')
    plt.ylabel('Özellik')
    plt.title(f'{model_type} Özellik Önemliliği', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(conf_matrix, class_names=None, figsize=(8, 6), cmap='Blues'):
    """
    Karmaşıklık matrisini görselleştirir
    
    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        Karmaşıklık matrisi
    class_names : list, default=None
        Sınıf isimleri
    figsize : tuple, default=(8, 6)
        Şekil boyutu
    cmap : str, default='Blues'
        Renk haritası
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    if class_names is None:
        class_names = ['Benign (0)', 'Malignant (1)']
    
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # Karmaşıklık matrisini çiz
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, 
                square=True, cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.title('Karmaşıklık Matrisi', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_roc_curve(fpr, tpr, auc, model_name='Model', figsize=(8, 6)):
    """
    ROC eğrisini çizer
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rate
    tpr : numpy.ndarray
        True positive rate
    auc : float
        Area under the curve
    model_name : str, default='Model'
        Model adı
    figsize : tuple, default=(8, 6)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # ROC eğrisini çiz
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_learning_curve(train_sizes, train_scores, test_scores, model_name='Model', figsize=(8, 6)):
    """
    Öğrenme eğrisini çizer
    
    Parameters:
    -----------
    train_sizes : numpy.ndarray
        Eğitim seti boyutları
    train_scores : numpy.ndarray
        Eğitim seti skorları
    test_scores : numpy.ndarray
        Test seti skorları
    model_name : str, default='Model'
        Model adı
    figsize : tuple, default=(8, 6)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ortalama ve standart sapma hesapla
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Eğitim eğrisini çiz
    plt.plot(train_sizes, train_mean, label='Eğitim Skoru', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    # Test eğrisini çiz
    plt.plot(train_sizes, test_mean, label='Çapraz Doğrulama Skoru', color='red', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.xlabel('Eğitim Örnekleri Sayısı')
    plt.ylabel('Doğruluk Oranı')
    plt.title(f'{model_name} Öğrenme Eğrisi', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_3d_scatter(data, features, target, figsize=(10, 8)):
    """
    3B dağılım grafiği çizer
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Veri seti
    features : list
        3 özellik adı
    target : str
        Hedef değişken adı
    figsize : tuple, default=(10, 8)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    if len(features) != 3:
        raise ValueError("features listesi tam olarak 3 özellik içermelidir.")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Hedef değişkene göre renklendirme
    target_values = data[target].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(target_values)))
    
    for target_value, color in zip(target_values, colors):
        subset = data[data[target] == target_value]
        ax.scatter(subset[features[0]], subset[features[1]], subset[features[2]], 
                  color=color, label=f'{target}={target_value}', alpha=0.7)
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title(f'3D Dağılım: {features[0]} vs {features[1]} vs {features[2]}', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    return fig

def plot_pca_components(X_pca, y, n_components=2, figsize=(10, 8)):
    """
    PCA bileşenlerini görselleştirir
    
    Parameters:
    -----------
    X_pca : numpy.ndarray
        PCA dönüşümü uygulanmış veri
    y : numpy.ndarray
        Hedef değişken
    n_components : int, default=2
        PCA bileşen sayısı
    figsize : tuple, default=(10, 8)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components 2 veya 3 olmalıdır.")
    
    # Benzersiz sınıfları ve renkleri tanımla
    classes = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    
    # 2D PCA grafiği
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, c in zip(classes, colors):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                        color=c, label=f'Sınıf {i}', alpha=0.7)
        
        plt.xlabel('Birinci Bileşen')
        plt.ylabel('İkinci Bileşen')
        
    # 3D PCA grafiği
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for i, c in zip(classes, colors):
            ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], 
                      color=c, label=f'Sınıf {i}', alpha=0.7)
        
        ax.set_xlabel('Birinci Bileşen')
        ax.set_ylabel('İkinci Bileşen')
        ax.set_zlabel('Üçüncü Bileşen')
    
    plt.title('PCA Bileşenleri', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    return fig

def plot_decision_boundary(X, y, model, feature_names, mesh_step_size=0.02, figsize=(10, 8)):
    """
    Karar sınırını çizer (2 özellik için)
    
    Parameters:
    -----------
    X : numpy.ndarray
        Özellik matrisi (2 özellikli)
    y : numpy.ndarray
        Hedef değişken
    model : sklearn model
        Eğitilmiş model
    feature_names : list
        Özellik adları (2 adet)
    mesh_step_size : float, default=0.02
        Izgara adım boyutu
    figsize : tuple, default=(10, 8)
        Şekil boyutu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Çizim nesnesi
    """
    if X.shape[1] != 2:
        raise ValueError("Bu fonksiyon yalnızca 2 boyutlu veri için çalışır.")
    
    # Şekil oluştur
    fig, ax = plt.subplots(figsize=figsize)
    
    # Izgara oluştur
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    
    # Karar fonksiyonunu hesapla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Karar sınırını çiz
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Karar Sınırı', fontsize=16)
    plt.tight_layout()
    
    return fig 