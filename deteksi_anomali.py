
import os
import time
import schedule
import pickle
import numpy as np
import pandas as pd
from collections import Counter, deque
from threading import Thread

# Coba impor library yang dibutuhkan
try:
    from scapy.all import sniff, TCP, UDP, IP
    from fpdf import FPDF
    import requests
    import matplotlib
    # === PERBAIKAN DI SINI ===
    # Set backend matplotlib ke 'Agg' SEBELUM mengimpor pyplot.
    # Ini adalah backend non-interaktif yang aman untuk thread.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"[ERROR KRITIS] Library yang dibutuhkan tidak ditemukan: {e}")
    print("Pastikan Anda telah menginstal 'scapy', 'fpdf', 'requests', dan 'matplotlib' dengan pip.")
    exit()

# --- [BAGIAN 1: KONFIGURASI DAN PERSIAPAN] ---

print("Memulai inisialisasi sistem deteksi anomali REAL-TIME dengan laporan PDF...")

# Konfigurasi Bot Telegram
TELEGRAM_BOT_TOKEN = "8059223693:AAHztCpxxeyp_bvCRgr5pwosjFgAOpJnNOk"
TELEGRAM_CHAT_ID = "1008901670"

# Konfigurasi Penangkapan Paket
NETWORK_INTERFACE = "eth3" # Ganti dengan nama interface Anda

# Nama file untuk model dan output
CNN_MODEL_FILE = 'cnn_feature_extractor.h5'
SVM_MODEL_FILE = 'svm_classifier.pkl'
PREPROCESSOR_FILE = 'nslkdd_preprocessor.pkl'
PDF_OUTPUT_FILE = 'Laporan_Aktivitas_Anomali_Periodik.pdf'
CHART_FILES = [] # Untuk melacak file grafik yang dibuat

# Variabel global untuk menyimpan data laporan sementara
laporan_sementara = {
    "total_paket": 0,
    "anomali_terdeteksi": 0,
    "detail_anomali": [] # Akan menyimpan dictionary untuk setiap anomali
}

# --- [BAGIAN 2: PEMUATAN MODEL] ---

print("\nMemuat model dan objek yang diperlukan untuk deteksi...")
if not all(os.path.exists(f) for f in [CNN_MODEL_FILE, SVM_MODEL_FILE, PREPROCESSOR_FILE]):
    print(f"[ERROR KRITIS] Salah satu file model tidak ditemukan. Jalankan skrip versi training terlebih dahulu.")
    exit()

try:
    from tensorflow.keras.models import load_model
    cnn_feature_extractor = load_model(CNN_MODEL_FILE, compile=False)
    with open(SVM_MODEL_FILE, 'rb') as f:
        svm_classifier = pickle.load(f)
    with open(PREPROCESSOR_FILE, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Semua model dan objek berhasil dimuat.")
except Exception as e:
    print(f"\n[ERROR KRITIS] Gagal memuat file model: {e}")
    exit()


# --- [BAGIAN 3: KELAS PROSESOR PAKET REAL-TIME] ---
# Kelas ini sama dengan versi 4.2, tidak ada perubahan di sini.
class RealTimePacketProcessor:
    def __init__(self):
        self.time_window_connections = deque()
        self.col_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]

    def get_service(self, dport):
        service_map = {80: 'http', 443: 'https', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain_u'}
        return service_map.get(dport, 'other')

    def get_flag(self, tcp_flags):
        if tcp_flags == 'S': return 'S0'
        if tcp_flags == 'R': return 'REJ'
        if 'R' in str(tcp_flags): return 'RSTO'
        if tcp_flags == 'F' or tcp_flags == 'FA' or tcp_flags == 'FPA': return 'SH'
        if tcp_flags == 'SA' or tcp_flags == 'SRA': return 'SF'
        return 'OTH'

    def process_packet(self, packet):
        global laporan_sementara
        laporan_sementara["total_paket"] += 1
        
        if not packet.haslayer(IP) or not (packet.haslayer(TCP) or packet.haslayer(UDP)): return

        current_time = time.time()
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = 'tcp' if packet.haslayer(TCP) else 'udp'
        dst_port = packet.dport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
        service = self.get_service(dst_port)
        src_bytes = len(packet.payload)
        flag = self.get_flag(packet[TCP].flags) if packet.haslayer(TCP) else 'OTH'

        while self.time_window_connections and current_time - self.time_window_connections[0]['timestamp'] > 2:
            self.time_window_connections.popleft()
        
        current_connection = {'timestamp': current_time, 'dst_ip': dst_ip, 'service': service, 'flag': flag}
        self.time_window_connections.append(current_connection)

        connections_to_dst = [conn for conn in self.time_window_connections if conn['dst_ip'] == dst_ip]
        count = len(connections_to_dst)
        connections_to_srv = [conn for conn in connections_to_dst if conn['service'] == service]
        srv_count = len(connections_to_srv)

        s_errors = sum(1 for conn in connections_to_dst if conn['flag'] in ['S0', 'S1', 'S2', 'S3'])
        r_errors = sum(1 for conn in connections_to_dst if conn['flag'] in ['REJ', 'RSTO', 'RSTR'])
        serror_rate = 0.0 if count == 0 else s_errors / count
        rerror_rate = 0.0 if count == 0 else r_errors / count
        srv_s_errors = sum(1 for conn in connections_to_srv if conn['flag'] in ['S0', 'S1', 'S2', 'S3'])
        srv_r_errors = sum(1 for conn in connections_to_srv if conn['flag'] in ['REJ', 'RSTO', 'RSTR'])
        srv_serror_rate = 0.0 if srv_count == 0 else srv_s_errors / srv_count
        srv_rerror_rate = 0.0 if srv_count == 0 else srv_r_errors / srv_count
        same_srv_rate = 1.0 if count == 0 else srv_count / count

        feature_dict = {
            'duration': 0, 'protocol_type': protocol, 'service': service, 'flag': flag, 'src_bytes': src_bytes, 
            'dst_bytes': 0, 'land': 1 if src_ip == dst_ip else 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 
            'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 
            'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 
            'is_host_login': 0, 'is_guest_login': 0, 'count': count, 'srv_count': srv_count, 'serror_rate': serror_rate, 
            'srv_serror_rate': srv_serror_rate, 'rerror_rate': rerror_rate, 'srv_rerror_rate': srv_rerror_rate, 
            'same_srv_rate': same_srv_rate, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': count, 
            'dst_host_srv_count': srv_count, 'dst_host_same_srv_rate': same_srv_rate, 'dst_host_diff_srv_rate': 0.0, 
            'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': serror_rate, 
            'dst_host_srv_serror_rate': srv_serror_rate, 'dst_host_rerror_rate': rerror_rate, 
            'dst_host_srv_rerror_rate': srv_rerror_rate
        }
        paket_df = pd.DataFrame([feature_dict], columns=self.col_names)

        try:
            hasil_prediksi = self.predict(paket_df)
            status = "ANOMALI" if hasil_prediksi == 1 else "Normal"
            if hasil_prediksi == 1:
                laporan_sementara["anomali_terdeteksi"] += 1
                laporan_sementara["detail_anomali"].append({
                    'timestamp': time.strftime('%H:%M:%S'), 'src_ip': src_ip, 'dst_ip': dst_ip, 
                    'service': service, 'flag': flag
                })
            print(f"[{time.strftime('%H:%M:%S')}] IP: {src_ip:<18} -> {dst_ip:<18} | Proto: {protocol:<4} | Flag: {flag:<4} | Status: {status:<8}", end='\r')
        except Exception as e:
            print(f"\n[ERROR PREDIKSI] Gagal memproses paket: {e}")

    def predict(self, paket_data_df):
        fitur_lengkap = preprocessor.transform(paket_data_df)
        fitur_reshaped = np.reshape(fitur_lengkap, (fitur_lengkap.shape[0], fitur_lengkap.shape[1], 1))
        fitur_dari_cnn = cnn_feature_extractor.predict(fitur_reshaped, verbose=0)
        prediksi_svm = svm_classifier.predict(fitur_dari_cnn)
        return prediksi_svm[0]

# --- [BAGIAN 4: FUNGSI PEMBUATAN DAN PENGIRIMAN LAPORAN PDF] ---

def create_bar_chart(data, title, filename, color='skyblue'):
    """Membuat dan menyimpan grafik batang dari data."""
    global CHART_FILES
    if not data:
        return
    
    labels = [str(item[0]) for item in data]
    values = [item[1] for item in data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color=color)
    plt.xlabel('Jumlah Deteksi')
    plt.title(title)
    plt.gca().invert_yaxis()  # Tampilkan nilai tertinggi di atas
    plt.tight_layout(pad=2.0)

    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', va='center')

    plt.savefig(filename)
    plt.close()
    CHART_FILES.append(filename) # Tambahkan ke daftar untuk dibersihkan nanti

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Laporan Aktivitas Anomali Jaringan Periodik', 0, 1, 'C')
        self.set_font('Arial', '', 9)
        self.cell(0, 5, f"Periode: 5 Menit Terakhir ({time.strftime('%Y-%m-%d %H:%M:%S')})", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

def buat_dan_kirim_laporan_pdf():
    global laporan_sementara, CHART_FILES
    print("\n--- MEMBUAT LAPORAN PDF PERIODIK ---")
    
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Ringkasan Aktivitas (Matriks Teks)', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"  - Total Paket Dianalisis: {laporan_sementara['total_paket']}", 0, 1)
    pdf.cell(0, 5, f"  - Total Prediksi Anomali: {laporan_sementara['anomali_terdeteksi']}", 0, 1)
    pdf.ln(5)

    if laporan_sementara['anomali_terdeteksi'] > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2. Visualisasi Grafik Anomali', 0, 1)
        
        top_ips = Counter(d['src_ip'] for d in laporan_sementara['detail_anomali']).most_common(5)
        top_services = Counter(d['service'] for d in laporan_sementara['detail_anomali']).most_common(5)
        top_flags = Counter(d['flag'] for d in laporan_sementara['detail_anomali']).most_common(5)

        create_bar_chart(top_ips, 'Top 5 IP Sumber Anomali', 'chart_ips.png', 'salmon')
        create_bar_chart(top_services, 'Top 5 Layanan Target Anomali', 'chart_services.png', 'lightblue')
        create_bar_chart(top_flags, 'Top 5 Flag TCP Anomali', 'chart_flags.png', 'lightgreen')
        
        if os.path.exists('chart_ips.png'):
            pdf.image('chart_ips.png', x=10, w=pdf.w - 20)
            pdf.ln(5)
        if os.path.exists('chart_services.png'):
            pdf.image('chart_services.png', x=10, w=pdf.w - 20)
            pdf.ln(5)
        if os.path.exists('chart_flags.png'):
            pdf.image('chart_flags.png', x=10, w=pdf.w - 20)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"3. Log Lengkap Anomali Terdeteksi ({len(laporan_sementara['detail_anomali'])} kasus)", 0, 1)
        pdf.set_font('Courier', 'B', 9)
        pdf.cell(25, 5, 'Waktu', 1, 0, 'C')
        pdf.cell(40, 5, 'IP Sumber', 1, 0, 'C')
        pdf.cell(40, 5, 'IP Tujuan', 1, 0, 'C')
        pdf.cell(20, 5, 'Layanan', 1, 0, 'C')
        pdf.cell(15, 5, 'Flag', 1, 1, 'C')
        pdf.set_font('Courier', '', 8)
        
        for anomali in laporan_sementara['detail_anomali']:
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font('Courier', 'B', 9)
                pdf.cell(25, 5, 'Waktu', 1, 0, 'C')
                pdf.cell(40, 5, 'IP Sumber', 1, 0, 'C')
                pdf.cell(40, 5, 'IP Tujuan', 1, 0, 'C')
                pdf.cell(20, 5, 'Layanan', 1, 0, 'C')
                pdf.cell(15, 5, 'Flag', 1, 1, 'C')
                pdf.set_font('Courier', '', 8)
            pdf.cell(25, 5, anomali['timestamp'], 1, 0)
            pdf.cell(40, 5, anomali['src_ip'], 1, 0)
            pdf.cell(40, 5, anomali['dst_ip'], 1, 0)
            pdf.cell(20, 5, anomali['service'], 1, 0)
            pdf.cell(15, 5, anomali['flag'], 1, 1)
    else:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Tidak ada anomali signifikan yang tercatat pada periode ini.', 0, 1)

    try:
        pdf.output(PDF_OUTPUT_FILE)
        print(f"Laporan PDF '{PDF_OUTPUT_FILE}' berhasil dibuat.")
        print("Mengirim laporan PDF ke Telegram...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(PDF_OUTPUT_FILE, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': f"Laporan Aktivitas Anomali Periodik - {time.strftime('%Y-%m-%d %H:%M:%S')}"}
            response = requests.post(url, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            print("Laporan PDF berhasil dikirim ke Telegram.")
        else:
            print(f"[ERROR] Gagal mengirim PDF. Status: {response.status_code}, Respon: {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat membuat atau mengirim PDF: {e}")
    finally:
        if os.path.exists(PDF_OUTPUT_FILE):
            os.remove(PDF_OUTPUT_FILE)
        for chart_file in CHART_FILES:
            if os.path.exists(chart_file):
                os.remove(chart_file)
        CHART_FILES = []
        laporan_sementara = {"total_paket": 0, "anomali_terdeteksi": 0, "detail_anomali": []}
        print("Statistik laporan dan file sementara telah direset.")

# --- [BAGIAN 5: FUNGSI UTAMA DAN PENJADWALAN] ---

def jalankan_penjadwal():
    schedule.every(5).minutes.do(buat_dan_kirim_laporan_pdf)
    while True:
        schedule.run_pending()
        time.sleep(1)

def main():
    try:
        scheduler_thread = Thread(target=jalankan_penjadwal, daemon=True)
        scheduler_thread.start()
        
        print(f"\n[INFO] Memulai penangkapan paket pada interface: {NETWORK_INTERFACE}")
        print("Sistem deteksi anomali sekarang berjalan. Tekan Ctrl+C untuk berhenti.")
        print("-" * 70)
        
        processor = RealTimePacketProcessor()
        sniff(iface=NETWORK_INTERFACE, prn=processor.process_packet, store=0)

    except PermissionError:
        print("\n[ERROR KRITIS] Gagal memulai penangkapan paket. Izin ditolak. Coba jalankan dengan 'sudo'.")
    except OSError as e:
        print(f"\n[ERROR KRITIS] Gagal memulai penangkapan pada interface '{NETWORK_INTERFACE}'. Detail: {e}")
    except KeyboardInterrupt:
        print("\n\n[INFO] Sistem dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n\n[ERROR KRITIS] Terjadi kesalahan tak terduga: {e}")

if __name__ == "__main__":
    main()
