import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Modülleri ekle
sys.path.append("..")
from src.data.data_processing import (
    load_data, save_data, clean_data, detect_outliers, handle_outliers, 
    select_features, preprocess_data
)

class TestDataProcessing(unittest.TestCase):
    """
    Veri işleme fonksiyonlarını test etmek için birim testler
    """
    
    def setUp(self):
        """
        Test verisini oluştur
        """
        # Test verisi oluştur
        np.random.seed(42)
        
        # Test için veri seti oluştur
        self.test_data = pd.DataFrame({
            'id': range(1, 101),
            'diagnosis': np.random.choice(['B', 'M'], size=100, p=[0.7, 0.3]),
            'feature1': np.random.normal(10, 2, 100),
            'feature2': np.random.normal(20, 5, 100),
            'feature3': np.random.normal(30, 3, 100)
        })
        
        # Bazı aykırı değerler ekle
        self.test_data.loc[0, 'feature1'] = 30  # Aykırı değer
        self.test_data.loc[1, 'feature2'] = 50  # Aykırı değer
        
        # Bazı eksik değerler ekle
        self.test_data.loc[2, 'feature1'] = np.nan
        self.test_data.loc[3, 'feature2'] = np.nan
        
        # Test dosyası yolu
        self.test_file = 'test_data.csv'
        
    def tearDown(self):
        """
        Test dosyalarını temizle
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_and_save_data(self):
        """
        load_data ve save_data fonksiyonlarını test et
        """
        # Veriyi kaydet
        save_result = save_data(self.test_data, self.test_file)
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(self.test_file))
        
        # Veriyi yükle
        loaded_data = load_data(self.test_file)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), len(self.test_data))
        
        # Olmayan dosyayı yüklemeyi dene
        invalid_data = load_data('invalid_file.csv')
        self.assertIsNone(invalid_data)
    
    def test_clean_data(self):
        """
        clean_data fonksiyonunu test et
        """
        # Veriyi temizle
        cleaned_data = clean_data(self.test_data)
        
        # ID sütununun kaldırıldığını kontrol et
        self.assertNotIn('id', cleaned_data.columns)
        
        # Eksik değerlerin doldurulduğunu kontrol et
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)
        
        # Hedef değişkenin dönüştürüldüğünü kontrol et
        self.assertTrue(all(isinstance(val, (int, np.int64)) for val in cleaned_data['diagnosis']))
        
        # Benign ve malignant değerlerinin doğru dönüştürüldüğünü kontrol et
        orig_benign_count = (self.test_data['diagnosis'] == 'B').sum()
        orig_malign_count = (self.test_data['diagnosis'] == 'M').sum()
        
        clean_benign_count = (cleaned_data['diagnosis'] == 0).sum()
        clean_malign_count = (cleaned_data['diagnosis'] == 1).sum()
        
        self.assertEqual(orig_benign_count, clean_benign_count)
        self.assertEqual(orig_malign_count, clean_malign_count)
    
    def test_detect_outliers(self):
        """
        detect_outliers fonksiyonunu test et
        """
        # Önce veriyi temizle (hedef değişkeni dönüştürmek için)
        clean_df = clean_data(self.test_data)
        
        # Z-score yöntemi ile aykırı değerleri tespit et
        outliers_zscore = detect_outliers(clean_df, method='zscore', threshold=3)
        
        # Aykırı değerlerin tespit edildiğini kontrol et
        self.assertTrue(len(outliers_zscore) > 0)
        
        # IQR yöntemi ile aykırı değerleri tespit et
        outliers_iqr = detect_outliers(clean_df, method='iqr')
        
        # Aykırı değerlerin tespit edildiğini kontrol et
        self.assertTrue(len(outliers_iqr) > 0)
        
        # Geçersiz yöntem ile aykırı değer tespitini dene
        with self.assertRaises(ValueError):
            detect_outliers(clean_df, method='invalid')
    
    def test_handle_outliers(self):
        """
        handle_outliers fonksiyonunu test et
        """
        # Önce veriyi temizle
        clean_df = clean_data(self.test_data)
        
        # Aykırı değerleri tespit et
        outliers = detect_outliers(clean_df, method='zscore', threshold=3)
        
        # Sınırlama yöntemi ile aykırı değerleri işle
        cap_df = handle_outliers(clean_df, outliers, method='cap')
        
        # Değiştirme yöntemi ile aykırı değerleri işle
        replace_df = handle_outliers(clean_df, outliers, method='replace')
        
        # Kaldırma yöntemi ile aykırı değerleri işle
        remove_df = handle_outliers(clean_df, outliers, method='remove')
        
        # İşlemlerin başarıyla gerçekleştiğini kontrol et
        self.assertEqual(len(clean_df), len(cap_df))
        self.assertEqual(len(clean_df), len(replace_df))
        self.assertLessEqual(len(remove_df), len(clean_df))
        
        # Geçersiz yöntem ile aykırı değer işlemeyi dene
        with self.assertRaises(ValueError):
            handle_outliers(clean_df, outliers, method='invalid')
    
    def test_select_features(self):
        """
        select_features fonksiyonunu test et
        """
        # Önce veriyi temizle
        clean_df = clean_data(self.test_data)
        
        # Özellikler ve hedef değişkeni ayır
        X = clean_df.drop('diagnosis', axis=1)
        y = clean_df['diagnosis']
        
        # SelectKBest yöntemi ile özellik seçimi
        X_select, indices = select_features(X, y, method='selectkbest', k=2)
        
        # Seçilen özellik sayısını kontrol et
        self.assertEqual(X_select.shape[1], 2)
        self.assertEqual(len(indices), 2)
        
        # PCA yöntemi ile özellik seçimi
        X_pca, _ = select_features(X, y, method='pca', k=2)
        
        # Seçilen özellik sayısını kontrol et
        self.assertEqual(X_pca.shape[1], 2)
        
        # Tüm özellikleri kullanma
        X_all, indices_all = select_features(X, y, method='all')
        
        # Tüm özelliklerin kullanıldığını kontrol et
        self.assertEqual(X_all.shape[1], X.shape[1])
        
        # Geçersiz yöntem ile özellik seçimini dene
        with self.assertRaises(ValueError):
            select_features(X, y, method='invalid')
    
    def test_preprocess_data(self):
        """
        preprocess_data fonksiyonunu test et
        """
        # Test verisi oluştur
        data = pd.DataFrame({
            'id': range(10),
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'diagnosis': ['B', 'M'] * 5
        })
        
        # preprocess_data fonksiyonunu çağır
        X, y, X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            data,
            scaling_method='standardization',
            feature_selection_method='selectkbest',
            n_features=2
        )
        
        # Sonuçları kontrol et
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertTrue(isinstance(scaler, (StandardScaler, MinMaxScaler)))

if __name__ == '__main__':
    unittest.main() 