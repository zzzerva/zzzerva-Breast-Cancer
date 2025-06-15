# Python 3.11.6 base image kullan (pytorch-tabnet uyumluluğu için)
FROM python:3.11.6-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle ve timeout ayarlarını yap
RUN pip install --no-cache-dir --upgrade pip && \
    pip config set global.timeout 1000 && \
    pip config set global.retries 10

# Önce requirements.txt'yi kopyala ve bağımlılıkları yükle
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
      --timeout 120 \
      --retries 5 \
      --resume-retries 2 \
      -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Port ayarı
EXPOSE 8501

# Uygulamayı çalıştır
CMD ["streamlit", "run", "applicate/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 