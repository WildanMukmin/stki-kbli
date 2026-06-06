from flask import Flask, render_template, request, abort
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

app = Flask(__name__)

# ==============================================================
# 1. Load Dataset
# ==============================================================
print("Memuat dataset KBLI...")
df = pd.read_csv('dataset.csv', dtype={'kode': str})
df['kode'] = df['kode'].astype(str).str.zfill(5) # formatnya 5 digit string

# Pastikan kolom 'kbli' ada (nama judul di notebook)
if 'kbli' not in df.columns and 'judul' in df.columns:
    df.rename(columns={'judul': 'kbli'}, inplace=True)

# ==============================================================
# 2. Inisialisasi Preprocessing (sesuai notebook)
# ==============================================================
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

def preprocess_text(teks):
    """Preprocessing identik dengan notebook Colab."""
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    teks = stopword_remover.remove(teks)
    teks = stemmer.stem(teks)
    return teks

# ==============================================================
# 3. Load atau Build TF-IDF (cache ke .pkl agar startup cepat)
# ==============================================================
TFIDF_MATRIX_PATH = 'tfidf_matrix.pkl'
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
CLEAN_COL = 'keterangan_bersih'

# ==============================================================
# Load Model TF-IDF
# ==============================================================
print("Memuat model TF-IDF...")

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix_kbli = pickle.load(f)

# buat kolom clean untuk matched_words
df['keterangan_bersih'] = (
    df['kbli'].fillna('') + ' ' + df['deskripsi'].fillna('')
).apply(preprocess_text)

print("✅ Model berhasil dimuat")

# ==============================================================
# ROUTING FLASK
# ==============================================================

# Page 1: Halaman Utama
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Page 2: Hasil Pencarian
@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return render_template('index.html', error="Input pencarian tidak boleh kosong!")

    # Ekstrak angka dari input pengguna sebelum teksnya dibersihkan 
    extracted_numbers = re.findall(r'\d+', user_query)
    
    # Preprocessing kueri (identik dengan notebook)
    clean_query = preprocess_text(user_query)
    query_words = set(clean_query.split())

    df_result = df.copy()
    
    df_result['score_tfidf'] = 0.0
    df_result['score_kode'] = 0.0

    # Hitung Skor Kemiripan TF-IDF (Teks) 
    if clean_query.strip():
        tfidf_matrix_query = vectorizer.transform([clean_query])
        similarity_scores = cosine_similarity(tfidf_matrix_query, tfidf_matrix_kbli).flatten()
        df_result['score_tfidf'] = similarity_scores
        
    # Hitung Skor Pencarian Berdasarkan Kode 
    if extracted_numbers:
        for num in extracted_numbers:
            # Beri skor 1.0 (100%) jika kodenya cocok persis
            exact_match = df_result['kode'] == num
            # Beri skor 0.85 (85%) jika kodenya diawali angka tersebut (partial match)
            partial_match = df_result['kode'].str.startswith(num)
            
            df_result.loc[exact_match, 'score_kode'] = 1.0
            df_result.loc[partial_match & ~exact_match, 'score_kode'] = 0.85

    # Ambil skor tertinggi antara skor Teks dan skor Kode 
    df_result['score'] = df_result[['score_tfidf', 'score_kode']].max(axis=1)
    
    top_results = (
        df_result[df_result['score'] > 0]
        .sort_values(by='score', ascending=False)
        .head(5)
    )

    results = []
    for _, row in top_results.iterrows():
        kbli_words = set(str(row[CLEAN_COL]).split())
        matched_tokens = query_words.intersection(kbli_words)

        # Menyesuaikan tampilan kata yang cocok (Matched Words) 
        matched_display = []
        if row['score_kode'] > 0 and extracted_numbers:
            matched_display.append(f"Kode: {', '.join(extracted_numbers)}")
        if matched_tokens:
            matched_display.append(f"Teks: {', '.join(matched_tokens)}")
            
        final_matched_text = ' | '.join(matched_display) if matched_display else 'Tidak ada'

        results.append({
            'kode': row['kode'],
            'judul': row['kbli'],
            'deskripsi': row['deskripsi'],
            'score': row['score'],
            'matched_words': final_matched_text,
        })

    return render_template('results.html', query=user_query, results=results)

# Page 3: Detail KBLI
@app.route('/kbli/<kode>', methods=['GET'])
def kbli_detail(kode):
    detail_data = df[df['kode'].astype(str).str.zfill(5) == str(kode)]
    if detail_data.empty:
        abort(404)
    selected_kbli = detail_data.iloc[0].to_dict()
    # Normalisasi key agar template konsisten
    selected_kbli['judul'] = selected_kbli.get('kbli', selected_kbli.get('judul', ''))
    return render_template('detail.html', kbli=selected_kbli)

if __name__ == '__main__':
    app.run(debug=True)
