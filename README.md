# Meme Kanseri Tahmin Uygulaması

Bu proje, Wisconsin Meme Kanseri (Diagnostic) veri seti üzerinde makine öğrenmesi ve derin öğrenme algoritmaları kullanarak meme kanseri tahmini yapan modern bir Streamlit uygulamasıdır.

## Proje Yapısı

```
Streamlit-analiz/
├── applicate/
│   ├── modules/           # Streamlit sayfa modülleri
│   └── app.py             # Ana uygulama dosyası
├── src/                   # Kaynak kodlar (veri işleme, görselleştirme)
│   └── data/
├── models/                # Model kodları ve kayıtları
│   ├── ml/                # Makine öğrenmesi modelleri
│   ├── dl/                # Derin öğrenme modelleri
│   └── saved_models/      # Kaydedilmiş modeller (keras, npy, joblib)
├── utils/                 # Yardımcı fonksiyonlar
├── tests/                 # Test dosyaları
│   └── tests/             # Test modülleri
├── data/                  # Örnek veri setleri
├── static/                # Statik dosyalar
├── requirements.txt       # Bağımlılıklar
├── Dockerfile             # Docker yapılandırması
├── docker-compose.yml     # Docker Compose
└── README.md
```

## Gereksinimler

- Python 3.11.6
- Docker (opsiyonel)
- Docker Compose (opsiyonel)

## Kurulum

1. Projeyi klonlayın:
```bash
git clone <repo-url>
cd Streamlit-analiz
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Uygulamayı başlatın:
```bash
streamlit run applicate/app.py
```

## Docker ile Çalıştırma

1. Docker image'ı oluşturun:
```bash
docker build -t meme-kanseri-tahmin .
```

2. Docker container'ı başlatın:
```bash
docker run -p 8501:8501 meme-kanseri-tahmin
```

3. Tarayıcıda açın:
```
http://localhost:8501
```

### Docker Compose ile
```bash
docker-compose up
```

## Uygulama Özellikleri

- **Veri Analizi:** Genel istatistikler, görselleştirme, özellik analizi
- **Veri Ön İşleme:** Eksik veri temizleme, aykırı değer tespiti, normalizasyon, özellik seçimi
- **Makine Öğrenmesi:** Lojistik Regresyon, Random Forest, SVM, Gradient Boosting, XGBoost
- **Derin Öğrenme:** MLP (Multi-Layer Perceptron) ile eğitim, değerlendirme, tahmin
- **Model Değerlendirme:** Doğruluk, Hassasiyet, Duyarlılık, F1, ROC AUC, Karışıklık Matrisi, SHAP Değerleri

## Test

Tüm testleri çalıştırmak için:
```bash
pytest tests/
```

- Makine öğrenmesi ve derin öğrenme için otomatik testler içerir.
- `tests/tests/test_dl_models.py` dosyasında temel MLP eğitimi ve tahmini test edilir.

## Kullanılan Teknolojiler

- Python 3.11.6
- Streamlit
- Scikit-learn
- TensorFlow/Keras
- SHAP
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır. 