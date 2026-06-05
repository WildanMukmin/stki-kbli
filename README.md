# 🔍 KBLI Search Engine

### Sistem Temu Kembali Informasi Klasifikasi Baku Lapangan Usaha Indonesia

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.3%2B-black?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TF--IDF-Algorithm-green?style=for-the-badge" />
</p>

---

## 📖 Tentang Proyek

**KBLI Search Engine** adalah aplikasi web berbasis Python yang dirancang untuk membantu pengguna menemukan kode **Klasifikasi Baku Lapangan Usaha Indonesia (KBLI)** yang paling relevan berdasarkan deskripsi bisnis atau kata kunci yang dimasukkan.

Aplikasi ini mengimplementasikan konsep **Sistem Temu Kembali Informasi (STKI / Information Retrieval System)** dengan memanfaatkan algoritma **TF-IDF (Term Frequency–Inverse Document Frequency)** dan **Cosine Similarity** untuk mengukur relevansi antara kueri pengguna dengan seluruh entri data KBLI.

> **Contoh Penggunaan:** Pengguna mengetik _"jual beli kopi dan teh secara online"_, sistem akan menampilkan kode KBLI yang paling relevan beserta skor kemiripannya.

---

## ✨ Fitur Utama

| Fitur                    | Deskripsi                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------- |
| 🔎 **Pencarian Cerdas**  | Menemukan kode KBLI berdasarkan kueri teks bebas (natural language query)          |
| 🤖 **Algoritma TF-IDF**  | Pembobotan kata yang akurat dengan mempertimbangkan frekuensi dan keunikan istilah |
| 📐 **Cosine Similarity** | Mengukur kemiripan vektor kueri dengan seluruh dokumen KBLI                        |
| ⚡ **Model Pre-trained** | Model TF-IDF disimpan dalam `.pkl` agar startup aplikasi lebih cepat               |
| 🌐 **Antarmuka Web**     | Tampilan bersih ala Google Search yang intuitif dan mudah digunakan                |
| 🏷️ **Kata Terkait**      | Menampilkan token yang cocok antara kueri dan dokumen hasil                        |
| 📄 **Detail KBLI**       | Halaman detail per kode KBLI dengan deskripsi lengkap cakupan lapangan usaha       |
| ☁️ **Siap Deploy**       | Dikonfigurasi dengan `Procfile` untuk deployment ke Heroku, Railway, atau Render   |

---

## 🗂️ Struktur Proyek

```
stki-kbli/
├── app.py                    # Entry point aplikasi Flask
├── dataset.csv               # Dataset KBLI (kode, judul, deskripsi)
├── KBLI_Preprocessed.csv     # Dataset KBLI hasil preprocessing
├── tfidf_vectorizer.pkl      # Model TF-IDF Vectorizer (pre-trained)
├── tfidf_matrix.pkl          # Matriks TF-IDF hasil transformasi dokumen
├── requirements.txt          # Daftar dependensi Python
├── Procfile                  # Konfigurasi deployment (Gunicorn)
├── .gitignore                # File/folder yang diabaikan Git
└── templates/
    ├── index.html            # Halaman utama (form pencarian)
    ├── results.html          # Halaman hasil pencarian
    └── detail.html           # Halaman detail per kode KBLI
```

---

## 🧠 Cara Kerja Sistem

```
Input Pengguna (Query)
        │
        ▼
  Preprocessing Teks
  ├── Lowercase
  ├── Hapus karakter non-alfabet
  ├── Stop Word Removal (PySastrawi)
  └── Stemming (PySastrawi)
        │
        ▼
  Transformasi TF-IDF
  (menggunakan vectorizer pre-trained)
        │
        ▼
  Cosine Similarity
  (query vector × TF-IDF matrix KBLI)
        │
        ▼
  Ranking & Filter Top-5 Hasil
        │
        ▼
  Tampilkan ke Pengguna
```

---

## 🛠️ Prasyarat

Pastikan perangkat lunak berikut telah terpasang:

- **Python** `3.7` atau lebih baru
- **pip** (Python Package Manager)

---

## 💻 Instalasi & Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone <url-repositori-anda>
cd stki-kbli
```

### 2. Buat Virtual Environment (Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python app.py
```

### 5. Akses di Browser

```
http://localhost:5000
```

---

## 📦 Dependensi

| Library         | Versi   | Kegunaan                                     |
| --------------- | ------- | -------------------------------------------- |
| `Flask`         | ≥ 2.3.0 | Web framework                                |
| `pandas`        | ≥ 2.0.0 | Manipulasi dataset CSV                       |
| `scikit-learn`  | ≥ 1.3.0 | TF-IDF & Cosine Similarity                   |
| `PySastrawi`    | ≥ 1.2.0 | Stemming & stopword removal Bahasa Indonesia |
| `gunicorn`      | latest  | WSGI server untuk deployment                 |
| `python-dotenv` | latest  | Manajemen environment variable               |

---

## ☁️ Deployment

Aplikasi ini siap untuk di-deploy ke berbagai platform cloud.

### Heroku / Railway / Render

File `Procfile` sudah dikonfigurasi untuk menjalankan server dengan **Gunicorn**:

```
web: gunicorn app:app
```

### Langkah Deployment ke Railway

```bash
# Login ke Railway CLI
railway login

# Inisialisasi proyek
railway init

# Deploy
railway up
```

> **Catatan:** Pastikan file `tfidf_vectorizer.pkl` dan `tfidf_matrix.pkl` ikut ter-push ke repository karena dibutuhkan saat runtime.

---

## 📸 Tampilan Aplikasi

| Halaman                     | Deskripsi                                                     |
| --------------------------- | ------------------------------------------------------------- |
| **Beranda** (`/`)           | Form pencarian bergaya Google dengan logo berwarna            |
| **Hasil** (`/search`)       | Daftar top-5 KBLI paling relevan dengan skor similaritas      |
| **Detail** (`/kbli/<kode>`) | Informasi lengkap satu kode KBLI beserta deskripsi cakupannya |

---

## 🔧 Konfigurasi Lanjutan

### Mengubah Jumlah Hasil Pencarian

Di `app.py`, ubah nilai pada `.head(5)` untuk menampilkan lebih banyak hasil:

```python
# Contoh: tampilkan 10 hasil teratas
top_results = (
    df_result[df_result['score'] > 0]
    .sort_values(by='score', ascending=False)
    .head(10)  # ← ubah di sini
)
```

### Meregenerasi Model TF-IDF

Jika dataset diperbarui, hapus file `.pkl` lama dan buat ulang model melalui notebook atau skrip preprocessing yang tersedia.

---

## 👨‍💻 Pengembang

Dibangunkan oleh **Wildan** sebagai proyek portofolio berbasis mata kuliah **Sistem Temu Kembali Informasi (STKI)**.

---

## 📄 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk keperluan akademis dan pembelajaran.
