import unittest
import numpy as np
import pandas as pd
import os
from models.dl.deep_learning import create_mlp_model, train_dl_model, save_dl_model, load_model, predict_dl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestDLModels(unittest.TestCase):
    def setUp(self):
        # Küçük bir örnek veri seti oluştur
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_mlp_training_and_prediction(self):
        # Modeli oluştur ve eğit
        model = create_mlp_model(input_dim=self.X_train.shape[1])
        history = model.fit(self.X_train, self.y_train, epochs=3, batch_size=8, verbose=0)
        # Tahmin
        preds = model.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])
        # Olasılıklar 0-1 aralığında olmalı
        self.assertTrue(np.all(preds >= 0) and np.all(preds <= 1))

    def test_save_and_load_model(self):
        model = create_mlp_model(input_dim=self.X_train.shape[1])
        model.fit(self.X_train, self.y_train, epochs=2, batch_size=8, verbose=0)
        model.save('mlp_test_model.keras')
        loaded = load_model('mlp_test_model.keras')
        preds1 = model.predict(self.X_test)
        preds2 = loaded.predict(self.X_test)
        np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
        os.remove('mlp_test_model.keras')

if __name__ == '__main__':
    unittest.main() 