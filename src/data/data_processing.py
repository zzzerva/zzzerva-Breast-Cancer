import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def load_data(file_path):
    """
    Veri setini yükler
    
    Parameters:
    -----------
    file_path : str
        Veri seti dosya yolu
        
    Returns:
    --------
    data : pandas.DataFrame
        Yüklenen veri seti
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Veri seti başarıyla yüklendi: {file_path}")
        data = data.dropna(axis=1, how='all')
        print(f"Tamamen boş olan sütunlar kaldırıldı: {data.columns}")
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        print(f"Unnamed sütunlar kaldırıldı: {data.columns}")
        return data
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

def save_data(data, file_path):
    """
    Veri setini kaydeder
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Kaydedilecek veri seti
    file_path : str
        Kayıt yolu
        
    Returns:
    --------
    success : bool
        İşlem başarısı
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Veri seti başarıyla kaydedildi: {file_path}")
        return True
    except Exception as e:
        print(f"Veri kaydedilirken hata oluştu: {e}")
        return False

def clean_data(data):
    """
    Veri temizleme işlemleri yapar
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Temizlenecek veri seti
        
    Returns:
    --------
    cleaned_data : pandas.DataFrame
        Temizlenmiş veri seti
    """
    # Kopya oluştur
    cleaned_data = data.copy()
    # ID sütunu varsa kaldır
    if 'id' in cleaned_data.columns:
        cleaned_data = cleaned_data.drop('id', axis=1)
        print("'id' sütunu kaldırıldı.")
    
    # Tamamen boş olan sütunları kaldır
    cleaned_data = cleaned_data.dropna(axis=1, how='all')
    # Eksik değerleri kontrol et ve işle
    missing_values = cleaned_data.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    if columns_with_missing:
        print(f"Eksik değer içeren sütunlar: {columns_with_missing}")
        print("Eksik değerler medyan ile doldurulacak.")
        
        for column in columns_with_missing:
            median_value = cleaned_data[column].median()
            cleaned_data[column].fillna(median_value, inplace=True)
    else:
        print("Veri setinde eksik değer yok.")
    
    # Hedef değişkeni sayısal değere dönüştür
    if 'diagnosis' in cleaned_data.columns and cleaned_data['diagnosis'].dtype == 'object':
        cleaned_data['diagnosis'] = cleaned_data['diagnosis'].map({'B': 0, 'M': 1})
        print("Hedef değişken ('diagnosis') sayısal değere dönüştürüldü (B=0, M=1).")
    
    return cleaned_data

def detect_outliers(data, method='zscore', threshold=3):
    """
    Aykırı değerleri tespit eder
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Aykırı değer tespiti yapılacak veri seti
    method : str, default='zscore'
        Kullanılacak yöntem ('zscore' veya 'iqr')
    threshold : float, default=3
        Eşik değeri (z-score için)
        
    Returns:
    --------
    outliers : dict
        Sütun bazında aykırı değer indeksleri
    """
    outliers = {}
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numeric_columns:
        # 'diagnosis' sütununu atla
        if column == 'diagnosis':
            continue
        
        if method == 'zscore':
            # Z-score yöntemi
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outlier_indices = np.where(z_scores > threshold)[0]
        elif method == 'iqr':
            # IQR (Interquartile Range) yöntemi
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_indices = np.where((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))[0]
        else:
            raise ValueError(f"Bilinmeyen yöntem: {method}. 'zscore' veya 'iqr' kullanın.")
        
        if len(outlier_indices) > 0:
            outliers[column] = outlier_indices
    
    return outliers

def handle_outliers(data, outliers, method='cap'):
    """
    Aykırı değerleri işler
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Aykırı değer işlemi yapılacak veri seti
    outliers : dict
        Sütun bazında aykırı değer indeksleri
    method : str, default='cap'
        Kullanılacak yöntem ('cap', 'remove' veya 'replace')
        
    Returns:
    --------
    processed_data : pandas.DataFrame
        İşlenmiş veri seti
    """
    processed_data = data.copy()
    
    for column, indices in outliers.items():
        if method == 'cap':
            # Sınırlama yöntemi (capping/winsorizing)
            Q1 = processed_data[column].quantile(0.25)
            Q3 = processed_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            processed_data[column] = processed_data[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"{column} sütunundaki aykırı değerler {lower_bound} ve {upper_bound} arasında sınırlandırıldı.")
            
        elif method == 'replace':
            # Medyan ile değiştirme
            median_value = processed_data[column].median()
            for idx in indices:
                processed_data.loc[idx, column] = median_value
            print(f"{column} sütunundaki {len(indices)} aykırı değer medyan ({median_value}) ile değiştirildi.")
            
        elif method == 'remove':
            # Satırları kaldırma (en son uygulanmalı)
            print(f"Uyarı: 'remove' yöntemi, tüm outlier işlemleri tamamlandıktan sonra tek seferde uygulanmalıdır.")
            
        else:
            raise ValueError(f"Bilinmeyen yöntem: {method}. 'cap', 'replace' veya 'remove' kullanın.")
    
    # Eğer 'remove' seçilmişse, tüm aykırı değer satırlarını kaldır
    if method == 'remove':
        all_indices = set()
        for indices in outliers.values():
            all_indices.update(indices)
        
        processed_data = processed_data.drop(index=list(all_indices))
        print(f"Toplam {len(all_indices)} satır (aykırı değerler) kaldırıldı.")
    
    return processed_data

def select_features(X, y, method='selectkbest', k=15):
    """
    Özellik seçimi yapar
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Özellik matrisi
    y : pandas.Series or numpy.ndarray
        Hedef değişken
    method : str, default='selectkbest'
        Kullanılacak yöntem ('selectkbest', 'pca' veya 'all')
    k : int, default=15
        Seçilecek özellik sayısı
        
    Returns:
    --------
    X_selected : numpy.ndarray
        Seçilen özellikler
    selected_indices : numpy.ndarray
        Seçilen özelliklerin indeksleri (sadece 'selectkbest' için)
    """
    if method == 'selectkbest':
        # SelectKBest yöntemi
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        if hasattr(X, 'columns'):
            selected_features = X.columns[selected_indices]
            print(f"Seçilen özellikler ({k}): {', '.join(selected_features)}")
        else:
            print(f"Seçilen özellik indeksleri ({k}): {selected_indices}")
        
        return X_selected, selected_indices
    
    elif method == 'pca':
        # PCA yöntemi
        pca = PCA(n_components=k)
        X_selected = pca.fit_transform(X)
        
        print(f"PCA ile {k} bileşen seçildi.")
        print(f"Açıklanan toplam varyans: {sum(pca.explained_variance_ratio_):.4f}")
        
        return X_selected, None
    
    elif method == 'all':
        # Tüm özellikleri kullan
        print("Tüm özellikler kullanılıyor.")
        return X, np.arange(X.shape[1])
    
    else:
        raise ValueError(f"Bilinmeyen yöntem: {method}. 'selectkbest', 'pca' veya 'all' kullanın.")

def preprocess_data(data, scaling_method='standardization', feature_selection_method='selectkbest', n_features=10, test_size=0.2, random_state=42):
    """
    Veri ön işleme adımlarını uygular
    
    Parameters:
    -----------
    data : pandas.DataFrame
        İşlenecek veri seti
    scaling_method : str, default='standardization'
        Ölçeklendirme yöntemi ('standardization' veya 'minmax')
    feature_selection_method : str, default='selectkbest'
        Özellik seçim yöntemi ('selectkbest', 'pca' veya 'all')
    n_features : int, default=10
        Seçilecek özellik sayısı
    test_size : float, default=0.2
        Test seti oranı
    random_state : int, default=42
        Rastgele durum değeri
        
    Returns:
    --------
    X : numpy.ndarray
        Özellik matrisi
    y : numpy.ndarray
        Hedef değişken
    X_train : numpy.ndarray
        Eğitim seti özellik matrisi
    X_test : numpy.ndarray
        Test seti özellik matrisi
    y_train : numpy.ndarray
        Eğitim seti hedef değişkeni
    y_test : numpy.ndarray
        Test seti hedef değişkeni
    features : list
        Seçilen özellik isimleri
    scaler : sklearn.preprocessing.Scaler
        Kullanılan ölçeklendirici
    """
    # Veriyi temizle
    cleaned_data = clean_data(data)
    
    # Hedef değişkeni ayır
    X = cleaned_data.drop('diagnosis', axis=1)
    y = cleaned_data['diagnosis']
    
    # Özellik seçimi
    if feature_selection_method != 'all':
        X, selected_indices = select_features(X, y, method=feature_selection_method, k=n_features)
        if hasattr(X, 'columns'):
            features = X.columns[selected_indices].tolist()
        else:
            features = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        features = X.columns.tolist()
    
    # Ölçeklendirme
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim-test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # y, y_train, y_test numpy array olsun
    y = np.asarray(y)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_scaled, y, X_train, X_test, y_train, y_test, features, scaler
