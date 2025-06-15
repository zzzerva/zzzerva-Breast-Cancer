import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Modülleri ekle
sys.path.append("..")
from models.ml.ml_models import (
    train_models, evaluate_models, get_cross_validation_scores,
    get_learning_curves, get_best_model, save_model, load_model
)

class TestMLModels(unittest.TestCase):
    """
    Makine öğrenmesi modellerini test etmek için birim testler
    """
    
    def setUp(self):
        """
        Test verisini oluştur
        """
        # Sentetik veri oluştur
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, 
            n_informative=5, random_state=42
        )
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Eğitim ve test setleri
        self.X_train = X_scaled[:160]
        self.X_test = X_scaled[160:]
        self.y_train = y[:160]
        self.y_test = y[160:]
        
        # Test için model kayıt yolu
        self.test_model_path = "test_model.pkl"
    
    def tearDown(self):
        """
        Test dosyalarını temizle
        """
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
    
    def test_train_models(self):
        """
        train_models fonksiyonunu test et
        """
        # Test verisi oluştur
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Modelleri eğit
        models_to_train = ["Lojistik Regresyon", "Random Forest"]
        models, training_times = train_models(X_train, y_train, models_to_train=models_to_train)
        
        # Sonuçları kontrol et
        self.assertIsInstance(models, dict)
        self.assertIsInstance(training_times, list)  # training_times artık bir liste
        self.assertTrue(all(isinstance(model, (LogisticRegression, RandomForestClassifier)) for model in models.values()))
        self.assertTrue(all(isinstance(time, dict) for time in training_times))  # Her bir eleman bir sözlük olmalı
        
        # Eğitim sürelerini göster
        for time_info in training_times:
            print(f"{time_info['model']} eğitim süresi: {time_info['accuracy']:.2f} saniye")
    
    def test_evaluate_models(self):
        """
        evaluate_models fonksiyonunu test et
        """
        # Modelleri eğit
        models_to_train = ["Lojistik Regresyon", "Random Forest"]
        trained_models, _ = train_models(
            self.X_train, self.y_train, 
            models_to_train=models_to_train, 
            random_state=42
        )
        
        # Modelleri değerlendir
        model_metrics = evaluate_models(trained_models, self.X_test, self.y_test)
        
        # Değerlendirme sonuçlarını kontrol et
        for model_name, metrics in model_metrics.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1', metrics)
            
            # Metrik değerlerinin 0 ile 1 arasında olduğunu kontrol et
            self.assertGreaterEqual(metrics['accuracy'], 0)
            self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_cross_validation(self):
        """
        get_cross_validation_scores fonksiyonunu test et
        """
        # Modelleri eğit
        models_to_train = ["Lojistik Regresyon"]
        trained_models, _ = train_models(
            self.X_train, self.y_train, 
            models_to_train=models_to_train, 
            random_state=42
        )
        
        # Çapraz doğrulama
        cv_scores = get_cross_validation_scores(
            trained_models, 
            np.vstack((self.X_train, self.X_test)), 
            np.concatenate((self.y_train, self.y_test)), 
            cv=5
        )
        
        # Çapraz doğrulama skorlarını kontrol et
        for model_name, scores in cv_scores.items():
            self.assertIn('scores', scores)
            self.assertIn('mean', scores)
            self.assertIn('std', scores)
            
            # Skor sayısını kontrol et
            self.assertEqual(len(scores['scores']), 5)  # cv=5
            
            # Ortalama skorun 0 ile 1 arasında olduğunu kontrol et
            self.assertGreaterEqual(scores['mean'], 0)
            self.assertLessEqual(scores['mean'], 1)
    
    def test_learning_curves(self):
        """
        get_learning_curves fonksiyonunu test et
        """
        # Modelleri eğit
        models_to_train = ["Lojistik Regresyon"]
        trained_models, _ = train_models(
            self.X_train, self.y_train, 
            models_to_train=models_to_train, 
            random_state=42
        )
        
        # Öğrenme eğrileri
        learning_curves_data = get_learning_curves(
            trained_models, 
            np.vstack((self.X_train, self.X_test)), 
            np.concatenate((self.y_train, self.y_test)), 
            cv=3,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        # Öğrenme eğrisi verilerini kontrol et
        for model_name, curve_data in learning_curves_data.items():
            self.assertIn('train_sizes', curve_data)
            self.assertIn('train_scores', curve_data)
            self.assertIn('test_scores', curve_data)
            
            # Boyutları kontrol et
            self.assertEqual(len(curve_data['train_sizes']), 5)  # train_sizes=np.linspace(0.1, 1.0, 5)
            self.assertEqual(curve_data['train_scores'].shape[0], 5)  # train_sizes sayısı
            self.assertEqual(curve_data['train_scores'].shape[1], 3)  # cv=3
    
    def test_get_best_model(self):
        """
        get_best_model fonksiyonunu test et
        """
        # Model metrikleri sözlüğü oluştur
        model_metrics = {
            "Model1": {
                "accuracy": 0.8,
                "precision": 0.7,
                "recall": 0.9,
                "f1": 0.8
            },
            "Model2": {
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85
            }
        }
        
        # F1 skoruna göre en iyi modeli bul
        best_model_name, best_score = get_best_model(model_metrics, metric='f1')
        
        # En iyi modelin "Model2" olduğunu kontrol et
        self.assertEqual(best_model_name, "Model2")
        self.assertEqual(best_score, 0.85)
        
        # Doğruluk skoruna göre en iyi modeli bul
        best_model_name, best_score = get_best_model(model_metrics, metric='accuracy')
        
        # En iyi modelin "Model2" olduğunu kontrol et
        self.assertEqual(best_model_name, "Model2")
        self.assertEqual(best_score, 0.9)
    
    def test_save_load_model(self):
        """
        save_model ve load_model fonksiyonlarını test et
        """
        # Modelleri eğit
        models_to_train = ["Lojistik Regresyon"]
        trained_models, _ = train_models(
            self.X_train, self.y_train, 
            models_to_train=models_to_train, 
            random_state=42
        )
        
        # İlk modeli al
        model_name = list(trained_models.keys())[0]
        model = trained_models[model_name]
        
        # Modeli kaydet
        save_result = save_model(model, model_name, self.test_model_path)
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(self.test_model_path))
        
        # Modeli yükle
        loaded_model = load_model(self.test_model_path)
        self.assertIsNotNone(loaded_model)
        
        # Orijinal ve yüklenen modellerin tahminlerinin aynı olduğunu kontrol et
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Var olmayan dosyadan model yüklemeyi dene
        invalid_model = load_model("invalid_model.pkl")
        self.assertIsNone(invalid_model)

if __name__ == '__main__':
    unittest.main() 