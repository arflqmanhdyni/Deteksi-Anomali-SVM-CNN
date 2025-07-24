
import os
import time
import schedule
import random
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# --- [BAGIAN 1: KONFIGURASI DAN PERSIAPAN] ---

print("Memulai inisialisasi sistem deteksi anomali...")

# Konfigurasi Bot Telegram
# GANTI DENGAN TOKEN DAN CHAT_ID ANDA YANG SEBENARNYA
#TELEGRAM_BOT_TOKEN = os.environ.get("8059223693:AAHztCpxxeyp_bvCRgr5pwosjFgAOpJnNOk")
#TELEGRAM_CHAT_ID = os.environ.get("1008901670")

TELEGRAM_BOT_TOKEN = "8059223693:AAHztCpxxeyp_b*****jFgAOpJnNOk"
TELEGRAM_CHAT_ID = "10089****70"
#TELEGRAM_TOKEN = "8059223693:AAHztCpxxeyp_bvCRgr5pwosjFgAOpJnNOk"
#TELEGRAM_CHAT_ID = "1008901670"
# Lokasi file dataset NSL-KDD
# GANTI DENGAN PATH FILE DATASET ANDA
PATH_TRAIN_DATA = 'KDDTrain+.txt'
PATH_TEST_DATA = 'KDDTest+.txt'

# Nama file untuk model dan preprocessor yang akan disimpan/dimuat
CNN_MODEL_FILE = 'cnn_feature_extractor.h5'
SVM_MODEL_FILE = 'svm_classifier.pkl'
PREPROCESSOR_FILE = 'nslkdd_preprocessor.pkl'

# Variabel global untuk menyimpan data laporan sementara
laporan_sementara = {
    "total_paket": 0,
    "anomali_terdeteksi": 0,
    "ip_sumber_anomali": []
}

# --- [BAGIAN 2: TRAINING MODEL MENGGUNAKAN NSL-KDD] ---
# Bagian ini hanya akan berjalan jika file model tidak ditemukan.

def latih_dan_simpan_model(path_train, path_test):
    """
    Fungsi lengkap untuk memuat, memproses, melatih, dan menyimpan model.
    """
    print("\n--- [FASE TRAINING DIMULAI] ---")
    print("File model tidak ditemukan. Memulai proses training dari awal...")

    # Impor library yang dibutuhkan untuk training
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.svm import SVC
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

    # 1. Memuat Dataset
    print(f"Membaca dataset dari {path_train} dan {path_test}...")
    col_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    df_train = pd.read_csv(path_train, header=None, names=col_names)
    df_test = pd.read_csv(path_test, header=None, names=col_names)

    # 2. Pra-pemrosesan Data
    print("Melakukan pra-pemrosesan data...")
    # Mengubah label menjadi biner (0 = normal, 1 = anomali)
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Memisahkan fitur dan label
    X_train = df_train.drop(['label', 'difficulty'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label', 'difficulty'], axis=1)
    y_test = df_test['label']

    # Mengidentifikasi fitur kategorikal dan numerik
    categorical_features = ['protocol_type', 'service', 'flag']
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Membuat transformer untuk encoding dan scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Menerapkan preprocessor ke data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Menyimpan preprocessor lengkap (scaler dan encoder)
    with open(PREPROCESSOR_FILE, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor lengkap telah disimpan ke {PREPROCESSOR_FILE}")

    # Menyesuaikan dimensi data untuk input CNN (Conv1D)
    X_train_cnn = np.reshape(X_train_transformed, (X_train_transformed.shape[0], X_train_transformed.shape[1], 1))
    X_test_cnn = np.reshape(X_test_transformed, (X_test_transformed.shape[0], X_test_transformed.shape[1], 1))

    # 3. Membangun dan Melatih Model CNN
    print("Membangun dan melatih model CNN...")
    input_shape = (X_train_cnn.shape[1], 1)

    input_layer = Input(shape=input_shape, name='input_layer')
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    feature_output = Dense(100, activation='relu', name='feature_layer')(x)
    classification_output = Dense(1, activation='sigmoid', name='classification_layer')(feature_output)

    cnn_model = Model(inputs=input_layer, outputs=classification_output)
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=128, validation_data=(X_test_cnn, y_test), verbose=1)

    # 4. Membuat Feature Extractor dan Melatih SVM
    print("Mengekstrak fitur dengan CNN dan melatih SVM...")
    feature_extractor = Model(inputs=cnn_model.inputs, outputs=cnn_model.get_layer('feature_layer').output)
    X_train_features = feature_extractor.predict(X_train_cnn)
    svm_classifier = SVC(kernel='rbf', gamma='scale', probability=True)
    svm_classifier.fit(X_train_features, y_train)

    # 5. Menyimpan Model
    print("Menyimpan model CNN feature extractor dan SVM classifier...")
    feature_extractor.save(CNN_MODEL_FILE)
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(svm_classifier, f)

    print(f"Model telah disimpan ke {CNN_MODEL_FILE} dan {SVM_MODEL_FILE}")
    print("--- [FASE TRAINING SELESAI] ---")

    # Mengembalikan data test (DataFrame asli) untuk digunakan di simulasi
    return df_test


# --- [BAGIAN 3: PEMUATAN ATAU TRAINING MODEL] ---

df_test_for_sim = None
if not all(os.path.exists(f) for f in [CNN_MODEL_FILE, SVM_MODEL_FILE, PREPROCESSOR_FILE]):
    if not os.path.exists(PATH_TRAIN_DATA) or not os.path.exists(PATH_TEST_DATA):
        print(f"\n[ERROR KRITIS] Dataset tidak ditemukan di {PATH_TRAIN_DATA} atau {PATH_TEST_DATA}.")
        print("Silakan unduh dataset NSL-KDD dan letakkan di direktori yang sama, atau ubah path di dalam skrip.")
        exit()
    df_test_for_sim = latih_dan_simpan_model(PATH_TRAIN_DATA, PATH_TEST_DATA)
else:
    print("Semua file model dan preprocessor ditemukan. Melewatkan fase training.")

print("\nMemuat model dan objek yang diperlukan untuk deteksi...")
try:
    from tensorflow.keras.models import load_model
    cnn_feature_extractor = load_model(CNN_MODEL_FILE)
    with open(SVM_MODEL_FILE, 'rb') as f:
        svm_classifier = pickle.load(f)
    with open(PREPROCESSOR_FILE, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Semua model dan objek berhasil dimuat.")
except Exception as e:
    print(f"\n[ERROR KRITIS] Gagal memuat file model: {e}")
    exit()

# Jika training tidak dijalankan, kita perlu memuat data test untuk simulasi
if df_test_for_sim is None:
    print("Memuat data test (DataFrame) untuk keperluan simulasi...")
    try:
        col_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        df_test_for_sim = pd.read_csv(PATH_TEST_DATA, header=None, names=col_names)
    except FileNotFoundError:
        print(f"[ERROR KRITIS] Tidak dapat menemukan {PATH_TEST_DATA} untuk menjalankan simulasi.")
        exit()


# --- [BAGIAN 4: MODUL UTAMA SISTEM (DETEKSI & PELAPORAN)] ---

def simulasikan_ekstraksi_fitur(data_sumber_df):
    """
    Generator yang mensimulasikan penangkapan paket.
    Mengambil baris acak dari DataFrame sumber (df_test) untuk simulasi.
    """
    while True:
        random_index = random.randint(0, len(data_sumber_df) - 1)
        # Ambil satu baris sebagai DataFrame untuk menjaga struktur kolom
        paket_df = data_sumber_df.iloc[random_index:random_index+1]

        # Pisahkan fitur dari label untuk prediksi
        fitur_paket_df = paket_df.drop(['label', 'difficulty'], axis=1, errors='ignore')

        ip_sumber = f"10.20.{random.randint(1, 254)}.{random.randint(1, 254)}"

        yield fitur_paket_df, ip_sumber
        time.sleep(random.uniform(0.01, 0.1))

def prediksi_anomali(paket_data_df):
    """
    Melakukan pipeline prediksi lengkap untuk satu paket data (DataFrame).
    """
    # 1. Pra-pemrosesan: Terapkan transformasi yang sama seperti saat training
    fitur_lengkap = preprocessor.transform(paket_data_df)

    # 2. Reshape data untuk input CNN
    fitur_reshaped = np.reshape(fitur_lengkap, (fitur_lengkap.shape[0], fitur_lengkap.shape[1], 1))

    # 3. Ekstraksi Fitur Lanjutan menggunakan CNN
    fitur_dari_cnn = cnn_feature_extractor.predict(fitur_reshaped, verbose=0)

    # 4. Klasifikasi Akhir menggunakan SVM
    prediksi_svm = svm_classifier.predict(fitur_dari_cnn)

    return prediksi_svm[0]

def kirim_laporan_telegram():
    """
    Mengumpulkan data dari `laporan_sementara`, memformat, dan mengirimkannya
    ke Telegram.
    """
    global laporan_sementara
    print("\n--- MEMPERSIAPKAN LAPORAN PERIODIK ---")

    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID":
        print("[PERINGATAN] Token atau Chat ID Telegram belum diatur. Laporan tidak akan dikirim.")
        laporan_sementara = {"total_paket": 0, "anomali_terdeteksi": 0, "ip_sumber_anomali": []}
        return

    counter_ip = Counter(laporan_sementara["ip_sumber_anomali"])
    top_5_ips = counter_ip.most_common(5)

    pesan = f"🚨 **Laporan Deteksi Anomali Jaringan** 🚨\n"
    pesan += f"*(Interval 5 Menit Terakhir)*\n\n"
    pesan += f"📦 **Total Lalu Lintas Dianalisis:** {laporan_sementara['total_paket']} paket\n"
    pesan += f"💥 **Anomali Terdeteksi:** {laporan_sementara['anomali_terdeteksi']} kali\n\n"

    if top_5_ips:
        pesan += "🔝 **Top 5 IP Sumber Anomali:**\n"
        for ip, count in top_5_ips:
            pesan += f"   - `{ip}` ({count} kali)\n"
    else:
        pesan += "✅ **Tidak ada anomali signifikan yang tercatat.**\n"
    pesan += f"\n_Laporan dibuat pada: {time.strftime('%Y-%m-%d %H:%M:%S')}_"

    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': pesan, 'parse_mode': 'Markdown'}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Laporan berhasil dikirim ke Telegram. Status: {response.status_code}")
        else:
            print(f"[ERROR] Gagal mengirim laporan. Status: {response.status_code}, Respon: {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat mengirim pesan Telegram: {e}")

    laporan_sementara = {"total_paket": 0, "anomali_terdeteksi": 0, "ip_sumber_anomali": []}


# --- [BAGIAN 5: LOOP UTAMA DAN PENJADWALAN] ---

def main():
    global laporan_sementara

    simulasi_generator = simulasikan_ekstraksi_fitur(df_test_for_sim)

    schedule.every(5).minutes.do(kirim_laporan_telegram)
    print("\n[INFO] Penjadwal laporan Telegram telah diaktifkan.")
    print("Sistem deteksi anomali sekarang berjalan. Tekan Ctrl+C untuk berhenti.")
    print("-" * 60)

    try:
        while True:
            fitur_df, ip = next(simulasi_generator)
            laporan_sementara["total_paket"] += 1

            hasil_prediksi = prediksi_anomali(fitur_df)

            if hasil_prediksi == 1:
                laporan_sementara["anomali_terdeteksi"] += 1
                laporan_sementara["ip_sumber_anomali"].append(ip)
                status = "ANOMALI"
            else:
                status = "Normal"

            print(f"[{time.strftime('%H:%M:%S')}] IP: {ip:<18} | Status: {status:<8}", end='\r')

            schedule.run_pending()

    except KeyboardInterrupt:
        print("\n\n[INFO] Sistem dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n\n[ERROR KRITIS] Terjadi kesalahan pada loop utama: {e}")

if __name__ == "__main__":
    main()
