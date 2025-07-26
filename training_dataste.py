# ---------------------------------------------------------------------------------
# IMPLEMENTASI LENGKAP: SISTEM DETEKSI ANOMALI (VERSI FINAL)
# ---------------------------------------------------------------------------------
# Skrip ini telah disempurnakan dengan fokus utama pada perbaikan overfitting CNN:
# 1. Menambahkan BatchNormalization untuk menstabilkan training.
# 2. Menambahkan L2 Kernel Regularization untuk mencegah bobot yang kompleks.
# 3. Meningkatkan Dropout rate untuk regularisasi yang lebih kuat.
# ---------------------------------------------------------------------------------

import os
import time
import schedule
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
# --- PERBAIKAN FINAL ---: Import tambahan untuk perbaikan CNN
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.under_sampling import RandomUnderSampler

# --- Konfigurasi Global ---
# Konfigurasi Telegram
TELEGRAM_BOT_TOKEN = "8059223693:AAHztCpxxeyp_bvCRgr5pwosjFgAOpJnNOk"
TELEGRAM_CHAT_ID = "1008901670"

# Path
PATH_TRAIN_DATA = 'KDDTrain+.txt'
PATH_TEST_DATA = 'KDDTest+.txt'
CNN_MODEL_FILE = 'cnn_feature_extractor_final.h5'
SVM_MODEL_FILE = 'svm_classifier_final.pkl'
PREPROCESSOR_FILE = 'nslkdd_preprocessor_final.pkl'

# Nama Kolom Dataset NSL-KDD
COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

# Variabel Global untuk Laporan
laporan_sementara = {
    "total_paket": 0,
    "anomali_terdeteksi": 0,
    "ip_sumber_anomali": []
}

def latih_dan_simpan_model(path_train, path_test):
    """Fungsi untuk melatih model CNN dan SVM, serta menyimpannya."""
    print("\n--- [FASE TRAINING DIMULAI (VERSI FINAL)] ---")

    # 1. Memuat Dataset
    print("1. Memuat dataset...")
    df_train = pd.read_csv(path_train, header=None, names=COL_NAMES)
    df_test = pd.read_csv(path_test, header=None, names=COL_NAMES)

    # 2. Pra-pemrosesan Label
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_train = df_train.drop(['label', 'difficulty'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label', 'difficulty'], axis=1)
    y_test = df_test['label']

    # 3. Pra-pemrosesan Fitur (Scaling dan Encoding)
    print("2. Melakukan pra-pemrosesan fitur...")
    categorical_features = ['protocol_type', 'service', 'flag']
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    with open(PREPROCESSOR_FILE, 'wb') as f:
        pickle.dump(preprocessor, f)

    # Menangani Class Imbalance dengan Random Undersampling
    print(f"3. Menangani class imbalance. Ukuran data sebelum sampling: {X_train_transformed.shape}")
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_transformed, y_train)
    print(f"   Ukuran data setelah sampling: {X_train_resampled.shape}")

    # 4. Mempersiapkan data untuk CNN
    X_train_cnn = np.reshape(X_train_resampled, (X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
    X_test_cnn = np.reshape(X_test_transformed, (X_test_transformed.shape[0], X_test_transformed.shape[1], 1))

    # 5. Membangun dan Melatih Model CNN dengan Regularisasi Kuat
    print("4. Membangun dan melatih model CNN dengan regularisasi kuat...")
    input_shape = (X_train_cnn.shape[1], 1)
    input_layer = Input(shape=input_shape, name='input_layer')
    
    # --- PERBAIKAN FINAL: Arsitektur CNN dengan Regularisasi Kuat ---
    # Blok Konvolusi 1
    x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x) # Dropout rate lebih tinggi

    # Blok Konvolusi 2
    x = Conv1D(128, 5, padding='same', kernel_regularizer=l2(0.001), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x) # Dropout rate lebih tinggi

    # Lapisan Akhir
    x = Flatten()(x)
    feature_output = Dense(100, activation='relu', name='feature_layer', kernel_regularizer=l2(0.001))(x)
    classification_output = Dense(1, activation='sigmoid', name='classification_layer')(feature_output)

    cnn_model = Model(inputs=input_layer, outputs=classification_output)
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callback untuk Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, restore_best_weights=True # Patience dinaikkan
    )

    history = cnn_model.fit(
        X_train_cnn, y_train_resampled,
        epochs=30, # Epoch ditambah karena ada regularisasi dan early stopping
        batch_size=128,
        validation_data=(X_test_cnn, y_test),
        callbacks=[early_stopping_callback]
    )

    # Simpan grafik akurasi dan loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('CNN Accuracy (Final)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN Loss (Final)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_training_performance_final.png")
    plt.close()

    # 6. Ekstraksi Fitur dan Melatih SVM
    print("5. Mengekstrak fitur dengan CNN dan melatih SVM...")
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('feature_layer').output)
    
    X_train_features = feature_extractor.predict(X_train_cnn)
    X_test_features = feature_extractor.predict(X_test_cnn)

    svm_classifier = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced')
    svm_classifier.fit(X_train_features, y_train_resampled)

    # 7. Evaluasi dan Simpan Model
    print("6. Mengevaluasi dan menyimpan model...")
    svm_preds = svm_classifier.predict(X_test_features)
    cm = confusion_matrix(y_test, svm_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomali"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix SVM (Final)")
    plt.savefig("svm_confusion_matrix_final.png")
    plt.close()

    # Simpan model
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(svm_classifier, f)
    feature_extractor.save(CNN_MODEL_FILE)

    # Kirim hasil ke Telegram
    kirim_gambar_ke_telegram("cnn_training_performance_final.png", "📊 *Grafik Performa CNN (Versi Final)*")
    kirim_gambar_ke_telegram("svm_confusion_matrix_final.png", "🧮 *Confusion Matrix SVM (Versi Final)*")

    return df_test

def kirim_gambar_ke_telegram(path_file, caption=""):
    """Mengirim file gambar ke chat Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(path_file, 'rb') as photo:
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
            files = {'photo': photo}
            response = requests.post(url, data=data, files=files, timeout=10)
            if response.status_code != 200:
                print(f"Gagal mengirim gambar ke Telegram: {response.text}")
    except Exception as e:
        print(f"Error saat mengirim gambar: {e}")

def prediksi_anomali(paket_df):
    """Memprediksi satu paket data apakah anomali atau tidak."""
    fitur_lengkap = preprocessor.transform(paket_df)
    fitur_reshaped = fitur_lengkap.reshape((fitur_lengkap.shape[0], fitur_lengkap.shape[1], 1))
    fitur_cnn = cnn_feature_extractor.predict(fitur_reshaped, verbose=0)
    return svm_classifier.predict(fitur_cnn)[0]

def simulasikan_ekstraksi_fitur(df):
    """Generator untuk mensimulasikan aliran paket data."""
    while True:
        idx = random.randint(0, len(df) - 1)
        baris = df.iloc[idx:idx+1]
        yield baris.drop(['label', 'difficulty'], axis=1, errors='ignore'), f"10.20.{random.randint(1,254)}.{random.randint(1,254)}"
        time.sleep(random.uniform(0.01, 0.1))

def kirim_laporan_telegram():
    """Mengirim laporan ringkasan anomali ke Telegram."""
    global laporan_sementara
    if laporan_sementara["total_paket"] == 0:
        return

    try:
        counter = Counter(laporan_sementara["ip_sumber_anomali"])
        top_ips = counter.most_common(5)
        
        msg = f"🚨 *Laporan Anomali Periodik*\n\n"
        msg += f"📦 Total paket dianalisis: *{laporan_sementara['total_paket']}*\n"
        msg += f"💥 Anomali terdeteksi: *{laporan_sementara['anomali_terdeteksi']}*\n\n"
        
        if top_ips:
            msg += "🔝 *Top 5 IP Sumber Anomali:*\n"
            for ip, count in top_ips:
                msg += f"- `{ip}` ({count} kali)\n"
        
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'},
            timeout=10
        )
    except Exception as e:
        print(f"Error saat mengirim laporan: {e}")
    finally:
        laporan_sementara = {"total_paket": 0, "anomali_terdeteksi": 0, "ip_sumber_anomali": []}

def main():
    """Fungsi utama untuk menjalankan sistem."""
    global cnn_feature_extractor, svm_classifier, preprocessor

    if not all(os.path.exists(f) for f in [CNN_MODEL_FILE, SVM_MODEL_FILE, PREPROCESSOR_FILE]):
        if not os.path.exists(PATH_TRAIN_DATA) or not os.path.exists(PATH_TEST_DATA):
            print(f"Dataset tidak ditemukan di {PATH_TRAIN_DATA} atau {PATH_TEST_DATA}.")
            print("Pastikan file dataset berada di folder yang sama dengan skrip.")
            return
        df_test = latih_dan_simpan_model(PATH_TRAIN_DATA, PATH_TEST_DATA)
    else:
        print("--- [MEMUAT MODEL YANG SUDAH ADA] ---")
        df_test = pd.read_csv(PATH_TEST_DATA, header=None, names=COL_NAMES)
        df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    cnn_feature_extractor = load_model(CNN_MODEL_FILE)
    with open(SVM_MODEL_FILE, 'rb') as f:
        svm_classifier = pickle.load(f)
    with open(PREPROCESSOR_FILE, 'rb') as f:
        preprocessor = pickle.load(f)

    print("\n--- [SIMULASI DETEKSI REAL-TIME DIMULAI] ---")
    print("Laporan akan dikirim ke Telegram setiap 5 menit.")
    print("Tekan Ctrl+C untuk menghentikan.")

    gen = simulasikan_ekstraksi_fitur(df_test)
    schedule.every(5).minutes.do(kirim_laporan_telegram)

    try:
        while True:
            df, ip = next(gen)
            laporan_sementara["total_paket"] += 1
            
            if prediksi_anomali(df) == 1:
                laporan_sementara["anomali_terdeteksi"] += 1
                laporan_sementara["ip_sumber_anomali"].append(ip)
            
            schedule.run_pending()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\nTerjadi error tak terduga: {e}")

if __name__ == '__main__':
    main()
