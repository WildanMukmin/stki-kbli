from flask import Flask, render_template, request, abort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

app = Flask(__name__)

# 1. Load Data Asli dan Data Preprocessed
print("Memuat dataset...")
df_asli = pd.read_csv('dataset.csv')
df_clean = pd.read_csv('kbli_preprocessed.csv') # Pastikan file ini ada di folder yang sama

# Pastikan kolom kode bertipe string agar tidak error saat digabung dan masuk ke URL
df_asli['kode'] = df_asli['kode'].astype(str).str.zfill(5)
df_clean['kode'] = df_clean['kode'].astype(str).str.zfill(5)

# Gabungkan kedua dataframe berdasarkan kolom 'kode'
# Hasilnya df akan memiliki 4 kolom: kode, judul, deskripsi, keterangan_bersih
df = pd.merge(df_asli, df_clean, on='kode', how='inner')

# Hindari error jika ada baris kosong pada data hasil preprocessing
df['keterangan_bersih'] = df['keterangan_bersih'].fillna('')

# 2. Inisialisasi Sastrawi Stemmer (HANYA untuk memproses query input pengguna)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_query(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return stemmer.stem(text)

# 3. Fit Model TF-IDF langsung menggunakan kolom data yang sudah dipreproses
print("Membuat model TF-IDF dari data keterangan_bersih...")
vectorizer = TfidfVectorizer()
tfidf_matrix_kbli = vectorizer.fit_transform(df['keterangan_bersih'])
print("Dataset dan model siap!")

# --- ROUTING FLASK ---

# Page 1: Search Box (Google Home)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Page 2: Serp (Google Search Result Page)
@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return render_template('index.html', error="Input pencarian tidak boleh kosong!")

    # Preproses input pencarian dari user
    clean_query = preprocess_query(user_query)
    query_words = set(clean_query.split())
    
    # Hitung kemiripan
    tfidf_matrix_query = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(tfidf_matrix_query, tfidf_matrix_kbli).flatten()
    
    df_result = df.copy()
    df_result['score'] = similarity_scores
    
    # Ambil 5 teratas
    top_results = df_result[df_result['score'] > 0].sort_values(by='score', ascending=False).head(8)
    
    results = []
    for idx, row in top_results.iterrows():
        # Lacak irisan kata menggunakan kolom 'keterangan_bersih'
        kbli_words = set(str(row['keterangan_bersih']).split())
        matched_tokens = query_words.intersection(kbli_words)
        
        results.append({
            'kode': row['kode'],
            'judul': row['judul'],
            'deskripsi': row['deskripsi'],
            'score': row['score'],
            'matched_words': ', '.join(matched_tokens) if matched_tokens else 'Tidak ada'
        })
    
    return render_template('results.html', query=user_query, results=results)

# Page 3: Detail Halaman KBLI saat Link diklik
@app.route('/kbli/<kode>', methods=['GET'])
def kbli_detail(kode):
    # Cari data di dataframe berdasarkan kode yang diklik
    detail_data = df[df['kode'] == str(kode)]
    
    if detail_data.empty:
        abort(404) # Jika kode tidak ditemukan tampilkan error 404
        
    # Ambil baris pertama data yang cocok
    selected_kbli = detail_data.iloc[0].to_dict()
    
    # Jika Anda ingin menampilkan teks bersihnya di detail.html
    selected_kbli['clean_features'] = selected_kbli.get('keterangan_bersih', '')
    
    return render_template('detail.html', kbli=selected_kbli)

if __name__ == '__main__':
    app.run(debug=True)