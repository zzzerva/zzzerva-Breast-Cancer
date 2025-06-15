import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import matplotlib.pyplot as plt

def create_mlp_model(input_dim, hidden_layers=[64, 32], dropout_rate=0.3):
    """MLP model oluştur"""
    model = Sequential([
        Dense(hidden_layers[0], activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden_layers[1], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_complex_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3, activation='elu'):
    """
    Daha karmaşık bir derin öğrenme modeli oluşturur
    
    Parameters:
    -----------
    input_dim : int
        Giriş boyutu
    hidden_layers : list, default=[128, 64, 32]
        Gizli katmanların nöron sayıları
    dropout_rate : float, default=0.3
        Dropout oranı
    activation : str, default='elu'
        Aktivasyon fonksiyonu
        
    Returns:
    --------
    model : tensorflow.keras.models.Model
        Oluşturulan model
    """
    inputs = Input(shape=(input_dim,))
    
    # İlk katman
    x = Dense(hidden_layers[0], activation=activation)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Gizli katmanlar
    for units in hidden_layers[1:]:
        x = Dense(units, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Çıkış katmanı
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Modeli derler
    
    Parameters:
    -----------
    model : tensorflow.keras.models.Sequential
        Derlenecek model
    learning_rate : float
        Öğrenme oranı
    
    Returns:
    --------
    model : tensorflow.keras.models.Sequential
        Derlenmiş model
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

def create_dl_model(input_dim):
    """Derin öğrenme modeli oluştur"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_tabnet_model(input_dim):
    """TabNet model oluştur"""
    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        momentum=0.3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        scheduler_params=dict(
            mode="min",
            patience=5,
            min_lr=1e-5,
            factor=0.9
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=10
    )
    return model

def train_dl_model(model_type, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    """Derin öğrenme modelini eğit"""
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # Model oluştur
    if model_type == 'MLP':
        model = create_mlp_model(X_train.shape[1])
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        # Model eğitimi
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return model, history, scaler
        
    elif model_type == 'TabNet':
        model = create_tabnet_model(X_train.shape[1])
        # Model eğitimi
        model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            eval_set=[(X_val_scaled, y_val)] if X_val is not None else None,
            max_epochs=epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        return model, None, scaler
    
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")

def save_dl_model(model, model_type, scaler):
    """Modeli ve scaler'ı kaydet"""
    os.makedirs('models/saved_models', exist_ok=True)
    
    if model_type == 'MLP':
        model.save(f'models/saved_models/{model_type.lower()}_model.keras')
    elif model_type == 'TabNet':
        model.save_model(f'models/saved_models/{model_type.lower()}_model.zip')
    
    np.save(f'models/saved_models/{model_type.lower()}_scaler.npy', scaler)

def load_dl_model(model_type):
    """Modeli ve scaler'ı yükle"""
    model_path = f'models/saved_models/{model_type.lower()}_model'
    scaler_path = f'models/saved_models/{model_type.lower()}_scaler.npy'
    
    if not os.path.exists(model_path + ('.keras' if model_type == 'MLP' else '.zip')):
        return None, None
    
    if model_type == 'MLP':
        model = load_model(model_path + '.keras')
    elif model_type == 'TabNet':
        model = TabNetClassifier()
        model.load_model(model_path + '.zip')
    
    scaler = np.load(scaler_path, allow_pickle=True).item()
    
    return model, scaler

def predict_dl(model, model_type, X, scaler):
    """Derin öğrenme modeli ile tahmin yap"""
    X_scaled = scaler.transform(X)
    
    if model_type == 'MLP':
        return model.predict(X_scaled)
    elif model_type == 'TabNet':
        return model.predict_proba(X_scaled)
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")

def evaluate_dl_model(model, X_test, y_test):
    """Derin öğrenme modelini değerlendir"""
    # Tahminler
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Metrikler
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['auc'] = auc(fpr, tpr)
    
    return metrics

def plot_training_history(history, figsize=(12, 4)):
    """Eğitim geçmişini görselleştir"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss grafiği
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy grafiği
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig 