# 🔍 KBLI-Search

Sistem Temu Kembali Informasi (STKI) untuk mencari **Klasifikasi Baku Lapangan Usaha Indonesia (KBLI 2020)** berdasarkan deskripsi bisnis menggunakan **TF-IDF + Cosine Similarity**.

## ✨ Fitur

- Pencarian semantik berbasis **TF-IDF + Cosine Similarity**
- **Preprocessing NLP** lengkap: lowercase, remove non-alpha, stopword removal, dan stemming menggunakan [Sastrawi](https://github.com/har07/PySastrawi)
- **Cache otomatis** model ke `.pkl` agar startup lebih cepat setelah pertama kali
- UI bersih bergaya Google Search (index, results, detail)
- Menampilkan skor relevansi & kata yang cocok

## 🗂️ Struktur Project

```
kbli-search/
├── app.py                  # Flask application
├── dataset.csv             # Dataset KBLI (kode, kbli, deskripsi)
├── requirements.txt
├── tfidf_matrix.pkl        # Auto-generated saat pertama run
├── tfidf_vectorizer.pkl    # Auto-generated saat pertama run
└── templates/
    ├── index.html          # Halaman utama
    ├── results.html        # Hasil pencarian
    └── detail.html         # Detail per kode KBLI
```

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/kbli-search.git
cd kbli-search
```

### 2. Buat virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan dataset

Letakkan file `dataset.csv` di root folder. Format kolom yang dibutuhkan:

| kode | kbli | deskripsi |
|------|------|-----------|
| 47111 | Perdagangan eceran berbagai... | Kelompok ini mencakup... |

> **Catatan:** Jika dataset Anda masih memiliki nama kolom `judul` (bukan `kbli`), app.py akan otomatis menyesuaikan.

### 5. Jalankan aplikasi

```bash
python app.py
```

Buka browser ke: `http://127.0.0.1:5000`

> Pada **pertama kali** dijalankan, sistem akan memproses seluruh dataset (stemming + TF-IDF). Proses ini mungkin memakan beberapa menit tergantung ukuran dataset. Hasil disimpan otomatis ke `tfidf_matrix.pkl` dan `tfidf_vectorizer.pkl` sehingga startup berikutnya akan jauh lebih cepat.

## 🧪 Contoh Kueri

- `usaha bengkel motor`
- `jualan baju online`
- `jasa konsultasi teknologi informasi`
- `budidaya ikan lele`
- `restoran dan rumah makan`

## 🔧 Teknologi

| Library | Fungsi |
|---------|--------|
| Flask | Web framework |
| Pandas | Manipulasi data |
| scikit-learn | TF-IDF Vectorizer & Cosine Similarity |
| PySastrawi | Stemming & stopword removal Bahasa Indonesia |

## 📓 Notebook

Proses training dan eksplorasi model tersedia di `Kbli.ipynb` (dilatih di Google Colab).

## 📄 Lisensi

MIT License
