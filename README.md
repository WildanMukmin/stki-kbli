```python
content = """# Pencarian KBLI (Sistem Temu Kembali Informasi)

Projek ini merupakan aplikasi web berasaskan Python yang dibangunkan untuk mencari dan memadankan **Klasifikasi Baku Lapangan Usaha Indonesia (KBLI)** menggunakan konsep Sistem Temu Kembali Informasi (STKI). Aplikasi ini menggunakan algoritma **TF-IDF (Term Frequency-Inverse Document Frequency)** dan **Cosine Similarity** untuk mencari kod KBLI yang paling relevan berdasarkan input teks pengguna.

## 🚀 Ciri-ciri Utama
- **Pencarian Pintar**: Mencari kod dan deskripsi KBLI berdasarkan kata kunci atau ayat (query).
- **Algoritma TF-IDF**: Pemprosesan teks yang cekap dan pantas menggunakan model TF-IDF yang telah dilatih dan disimpan (`tfidf_vectorizer.pkl` & `tfidf_matrix.pkl`).
- **Antaramuka Web Interaktif**: Dilengkapi dengan halaman carian utama (`index.html`), senarai hasil carian (`results.html`), dan butiran terperinci KBLI (`detail.html`).
- **Sedia untuk Deployment**: Mengandungi fail `Procfile` untuk kemudahan pelancaran ke platform awan (cloud) seperti Heroku atau Railway.

## 📂 Struktur Fail dan Direktori
- `app.py`: Fail utama aplikasi web (kemungkinan besar menggunakan framework Flask).
- `templates/`: Direktori yang mengandungi fail antaramuka HTML pengguna (`index.html`, `results.html`, `detail.html`).
- `kbli.csv` / `dataset.csv` / `KBLI_Preprocessed.csv`: Fail dataset yang mengandungi pangkalan data maklumat KBLI.
- `tfidf_matrix.pkl` & `tfidf_vectorizer.pkl`: Model TF-IDF pra-latih (pre-trained) yang telah diekstrak untuk mempercepatkan proses carian.
- `requirements.txt`: Senarai dependensi dan pustaka (libraries) Python yang diperlukan untuk menjalankan projek ini.
- `Procfile`: Fail konfigurasi khusus untuk proses *deployment*.

## 🛠️ Keperluan Sistem (Prerequisites)
Pastikan perisian berikut telah dipasang pada komputer anda sebelum memulakan projek:
- Python 3.7 atau ke atas
- `pip` (Pengurus Pakej Python)

## 💻 Cara Pemasangan & Menjalankan Aplikasi (Installation & Setup)

1. **Klon Repositori (Clone Repository)**

```

````text
File README.md successfully created.

```bash
   git clone <url-repositori-anda>
   cd stki-kbli

````

2. **Pasang Dependensi (Install Dependencies)**
   Adalah amat disyorkan untuk menggunakan persekitaran maya (virtual environment) Python.

```bash
pip install -r requirements.txt

```

3. **Jalankan Aplikasi (Run the Application)**

```bash
python app.py

```

4. **Akses Aplikasi**
   Buka pelayar web (web browser) pilihan anda dan layari pautan berikut:
   `http://localhost:5000` (atau port lain yang ditetapkan di dalam terminal).

## ☁️ Deployment (Pelancaran)

Aplikasi ini sedia untuk dilancarkan terus ke platform pengehosan (hosting) seperti Heroku, Railway, atau Render. Fail `Procfile` telah disediakan untuk menetapkan arahan permulaan pelayan web (web server).

## 👨‍💻 Pembangun

Dibangunkan oleh **Wildan** sebagai sebahagian daripada projek portofolio atau tugasan berasaskan Sistem Temu Kembali Informasi (STKI).
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(content)

print("File README.md successfully created.")

```

```
